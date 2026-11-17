# Samsung Galaxy S24 Ultra / Termux

The supplied GGUF artifacts are model data, not APK files. The cloud build produced Linux x86-64 binaries; they are **not** Android ARM binaries.

For local execution, use a current Termux installation and compile llama.cpp for Android ARM64:

```bash
pkg update
pkg install git clang cmake ninja
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build llama.cpp/build --target llama-cli -j 4
termux-setup-storage
./llama.cpp/build/bin/llama-cli -m /sdcard/Download/aetheris-qwen3-0.6b-research-Q4_K_M.gguf -c 2048 -t 4 -n 128 -st -p 'Napisz po polsku krótką scenę fikcyjną.'
```

Copy the completed Q4_K_M file to `/sdcard/Download/` beforehand. Verify its published SHA256 before executing it. A short context is recommended on mobile until a device-specific benchmark is available. 12 GB physical device RAM does not imply 12 GB available to the process. Do not claim that this binary or workflow was tested on the actual S24 Ultra.

The large 8B candidate is not included. Keep metadata/consent policy enforcement outside any abliterated model.
