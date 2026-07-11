cask "mlx-core" do
  version "26.7.5"
  sha256 "9c820b522afdeb6dd69390b7c2533225f6d5d9282eec9a5020e5fe4bcadc27c4"

  url "https://github.com/ddalcu/mlx-serve/releases/download/v#{version}/MLXCore.dmg"
  name "MLX Core"
  desc "Native LLM server for Apple Silicon with OpenAI & Anthropic compatible APIs"
  homepage "https://github.com/ddalcu/mlx-serve"

  depends_on macos: :tahoe
  depends_on arch: :arm64

  app "MLX Core.app"

  zap trash: [
    "~/.mlx-serve",
  ]
end
