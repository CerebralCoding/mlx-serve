import Foundation

/// The lifecycle of one hands-free voice turn, modeled as a pure state machine
/// so the wiring in `VoiceModeController` is trivially testable. The controller
/// feeds hardware/stream events in; the machine decides the next UI state.
enum VoiceTurnState: Equatable {
    case idle          // voice mode closed / not started
    case listening     // mic open, waiting for the user to start talking
    case recognizing   // user is talking, partial transcript updating
    case thinking      // transcript submitted, model generating (nothing audible yet)
    case speaking      // synthesizer reading the answer aloud
    case error(String)
}

enum VoiceTurnEvent: Equatable {
    case start               // user opened voice mode
    case speechStarted       // VAD/recognizer detected the user talking
    case transcriptFinalized // silence endpointing produced a final transcript → submit
    case responseStarted     // first speakable token of the answer arrived
    case turnFinished        // answer fully generated AND the TTS queue drained
    case bargeIn             // user interrupted while the assistant was speaking
    case utteranceDismissed  // heard speech, but it wasn't addressed to the assistant (no wake word)
    case failed(String)      // recoverable error surfaced to the orb
    case stop                // user closed voice mode
}

enum VoiceTurnMachine {
    static func reduce(_ state: VoiceTurnState, on event: VoiceTurnEvent) -> VoiceTurnState {
        // Universal exits — closing the mode or a failure win from any state.
        switch event {
        case .stop:           return .idle
        case .failed(let m):  return .error(m)
        default:              break
        }

        switch (state, event) {
        case (.idle, .start):                       return .listening

        case (.listening, .speechStarted):          return .recognizing
        case (.listening, .transcriptFinalized):    return .thinking   // utterance too short to trip VAD-start
        case (.recognizing, .transcriptFinalized):  return .thinking
        case (.recognizing, .bargeIn):              return .listening  // user restarted mid-utterance
        case (.recognizing, .utteranceDismissed):   return .listening  // not addressed to us → keep listening
        case (.listening, .utteranceDismissed):     return .listening

        case (.thinking, .responseStarted):         return .speaking
        case (.thinking, .turnFinished):            return .listening  // model produced nothing speakable
        case (.thinking, .bargeIn):                 return .listening  // user cut the turn off before it spoke

        case (.speaking, .turnFinished):            return .listening
        case (.speaking, .bargeIn):                 return .listening  // user interrupted the assistant

        default:                                    return state
        }
    }
}
