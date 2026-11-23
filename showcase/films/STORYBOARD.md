# Forge short films

| Film | Beats | First-party evidence |
| --- | --- | --- |
| Fusion | Run sample → separate scratch passes → fused register flow → memory and latency | Verified output and plans, Linux process RSS samples, release E5 native timing |
| Lanes | Eight elements → add a ninth → expand to seventeen | Native AVX2 JIT disassembly and bitwise-checked outputs for all three shapes |
| Trade-offs | Float32 order → tile edges → matched baselines → cache behavior | Independent reference outputs, pinned Python medians, actual cache counters |

The composition generator uses ElevenLabs takes for the spoken scenes. Scene boundaries come from audio duration and a short reading hold. Unvoiced scenes have a fixed reading hold for the diagrams and measurements. A quiet crossfade bridges beats. The main panel enters after the headline. Each pass highlights its operations and scratch boundaries, the packed groups highlight in execution order, and the RSS bars fill to their measured totals. Plans, opcodes, outputs, and timings originate in Forge’s capture JSON and release evidence ZIP. No third-party reference imagery is used. The WebVTT files and script accompany the MP4s.
