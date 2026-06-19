cask "mlx-core" do
  version "26.6.11"
  sha256 "ea310582dddb006aa1373a4e45ca91a6983620bd8d83be60bfa99eab3b76bdfa"

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
