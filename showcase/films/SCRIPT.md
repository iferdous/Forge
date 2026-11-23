# Film transcripts

The videos use ElevenLabs narration by Liam on the spoken scenes. Several scenes use on-screen evidence and captions to give the viewer time to read the numbers and instructions.

## One loop instead of three

1. Here’s the whole idea behind Forge. This Python expression adds two, multiplies by three, and takes the square root. We’ll run the same nine numbers two ways and compare what the CPU has to do.
2. Start with fusion turned off. Forge finishes the add for every input and saves that array. Then it reads that array back for the multiply. The square root does the same thing, so there are three passes and two intermediate arrays.
3. Turn fusion on and look at the new loop. Forge reads a group, does all three operations while those values are in CPU registers, and writes the final answers. The output stays the same. Those two temporary arrays disappear.
4. That difference shows up at larger sizes. The memory test saw about sixty-four mebibytes of extra resident memory on the first unfused call, versus about one page when fused. In a separate timing test, the fused kernel took about thirteen microseconds, down from twenty-three.

## Eight lanes and a tail

1. How can a CPU work on eight numbers at once? Forge emits a vector loop. In this capture, the instructions read eight thirty-two-bit floats into one wide register, then add, multiply, and take their square roots together.
2. Now add one more input. The first eight still go through the vector loop. Watch the last box: the ninth value goes through the one-at-a-time cleanup path, and its answer is three.
3. With seventeen inputs, there are two groups of eight and one leftover. The loop handles both groups before the final scalar step. Forge checks the result against its reference, including that last value.

## Where Forge wins and loses

1. A compiler has to keep the math predictable. Add one to sixteen million seven hundred seventy-seven thousand two hundred sixteen in thirty-two-bit floating point, and it rounds away. That’s why this ordered sum returns four.
2. Matrix multiply has a different problem: reusing data. Forge works on blocks instead of walking the full matrices in one sweep. This example crosses the tile boundaries, so the kernel has to finish with smaller blocks along the edges.
3. The benchmarks show where each approach pays off. Forge wins on this short chain because it combines three operations into one pass. For the larger matrix, NumPy’s tuned math library is faster, even with a single thread.
4. There’s also a cost to generating machine code in the first place. Repeat the same shape and Forge reuses that compiled code. Change the length, and the cache records a miss because the kernel needs to be compiled for the new shape.
