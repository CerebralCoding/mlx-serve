cask "mlx-core" do
  version "26.6.1"
  sha256 "141a7800d875d8e27f26c8b0dabe6644b2f12bb2f69b1509449fc9f668ad4a97"

  url "https://github.com/ddalcu/mlx-serve/releases/download/v#{version}/MLXCore.dmg"
  name "MLX Core"
  desc "Native LLM server for Apple Silicon with OpenAI & Anthropic compatible APIs"
  homepage "https://github.com/ddalcu/mlx-serve"

  depends_on macos: ">= :sonoma"
  depends_on arch: :arm64

  app "MLX Core.app"

  zap trash: [
    "~/.mlx-serve",
  ]
end
