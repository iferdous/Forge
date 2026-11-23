# Film transcripts

## One loop instead of three

1. Forge turns this Python expression into a native kernel. Nine elements go in, and both execution plans return the same float32 values.
2. The unfused plan runs add, multiply, and square root in separate loops. Two temporary arrays carry values between those passes.
3. The fused plan performs all three operations inside one loop. The intermediate values stay in registers until the result is written.
4. In the isolated Linux probe, the unfused workspace needs sixty-four MiB of scratch. For the separate sixty-five thousand five hundred thirty-six element case, compare the warm native A V X two medians.

## Eight lanes and a tail

1. Eight inputs fill one AVX2 vector. The actual JIT object shows packed add, multiply, and square root instructions operating on Y M M registers.
2. Add a ninth input and it takes the scalar cleanup path. The last output still matches the float32 reference bit for bit.
3. With seventeen inputs, the same schedule runs two packed iterations and one scalar iteration. The packed body increments the index by eight.

## Where Forge wins and loses

1. An ordered float32 sum loses the one when it adds it to two to the twenty-fourth power. Forge preserves that order and returns four.
2. The matrix kernel tiles rows, columns, and the reduction dimension. This seventeen by thirty-three by thirty-five case needs an edge tile on all three axes.
3. The fused pointwise chain can beat NumPy. For the five hundred and twelve square matrix multiply, NumPy wins against Forge even with one BLAS thread.
4. Repeating a cache-enabled request with the same shape produces a hit. Changing the length from four to nine produces a fresh cache miss.
