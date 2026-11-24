
forge.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0: 48 89 f8                     	movq	%rdi, %rax
       3: 48 8b 0e                     	movq	(%rsi), %rcx
       6: 48 8b 3a                     	movq	(%rdx), %rdi
       9: c5 fa 10 05 00 00 00 00      	vmovss	(%rip), %xmm0           # 0x11 <forge_kernel+0x11>
      11: c5 fa 10 0d 00 00 00 00      	vmovss	(%rip), %xmm1           # 0x19 <forge_kernel+0x19>
      19: c5 fa 10 15 00 00 00 00      	vmovss	(%rip), %xmm2           # 0x21 <forge_kernel+0x21>
      21: 48 89 c6                     	movq	%rax, %rsi
      24: 48 89 ca                     	movq	%rcx, %rdx
      27: eb 07                        	jmp	0x30 <forge_fused_10>
      29: 0f 1f 80 00 00 00 00         	nopl	(%rax)

0000000000000030 <forge_fused_10>:
      30: 48 85 f6                     	testq	%rsi, %rsi
      33: 74 6a                        	je	0x9f <forge_fused_10+0x6f>
      35: 48 83 fe 08                  	cmpq	$0x8, %rsi
      39: 73 04                        	jae	0x3f <forge_fused_10+0xf>
      3b: 31 c0                        	xorl	%eax, %eax
      3d: eb 42                        	jmp	0x81 <forge_fused_10+0x51>
      3f: 48 89 f0                     	movq	%rsi, %rax
      42: 48 83 e0 f8                  	andq	$-0x8, %rax
      46: c4 e2 7d 18 d8               	vbroadcastss	%xmm0, %ymm3
      4b: c4 e2 7d 18 e1               	vbroadcastss	%xmm1, %ymm4
      50: c4 e2 7d 18 ea               	vbroadcastss	%xmm2, %ymm5
      55: 31 c9                        	xorl	%ecx, %ecx
      57: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)
      60: c5 e4 59 34 8a               	vmulps	(%rdx,%rcx,4), %ymm3, %ymm6
      65: c5 dc 58 f6                  	vaddps	%ymm6, %ymm4, %ymm6
      69: c5 d4 59 f6                  	vmulps	%ymm6, %ymm5, %ymm6
      6d: c5 fc 51 f6                  	vsqrtps	%ymm6, %ymm6
      71: c5 fc 11 34 8f               	vmovups	%ymm6, (%rdi,%rcx,4)
      76: 48 83 c1 08                  	addq	$0x8, %rcx
      7a: 48 39 c8                     	cmpq	%rcx, %rax
      7d: 75 e1                        	jne	0x60 <forge_fused_10+0x30>
      7f: eb 19                        	jmp	0x9a <forge_fused_10+0x6a>
      81: c5 fa 59 1c 82               	vmulss	(%rdx,%rax,4), %xmm0, %xmm3
      86: c5 f2 58 db                  	vaddss	%xmm3, %xmm1, %xmm3
      8a: c5 ea 59 db                  	vmulss	%xmm3, %xmm2, %xmm3
      8e: c5 e2 51 db                  	vsqrtss	%xmm3, %xmm3, %xmm3
      92: c5 fa 11 1c 87               	vmovss	%xmm3, (%rdi,%rax,4)
      97: 48 ff c0                     	incq	%rax
      9a: 48 39 c6                     	cmpq	%rax, %rsi
      9d: 75 e2                        	jne	0x81 <forge_fused_10+0x51>
      9f: c5 f8 77                     	vzeroupper
      a2: c3                           	retq
