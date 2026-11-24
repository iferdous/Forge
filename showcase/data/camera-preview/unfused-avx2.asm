
unfused.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0: 41 57                        	pushq	%r15
       2: 41 56                        	pushq	%r14
       4: 41 54                        	pushq	%r12
       6: 53                           	pushq	%rbx
       7: 50                           	pushq	%rax
       8: 49 89 d6                     	movq	%rdx, %r14
       b: 48 89 fb                     	movq	%rdi, %rbx
       e: 48 8b 16                     	movq	(%rsi), %rdx
      11: 4d 8b 7e 08                  	movq	0x8(%r14), %r15
      15: c5 fa 10 05 00 00 00 00      	vmovss	(%rip), %xmm0           # 0x1d <forge_kernel+0x1d>
      1d: 4c 89 ff                     	movq	%r15, %rdi
      20: 48 89 de                     	movq	%rbx, %rsi
      23: e8 58 00 00 00               	callq	0x80 <forge_multiply_3>
      28: 4d 8b 66 10                  	movq	0x10(%r14), %r12
      2c: c5 fa 10 05 00 00 00 00      	vmovss	(%rip), %xmm0           # 0x34 <forge_kernel+0x34>
      34: 4c 89 e7                     	movq	%r12, %rdi
      37: 48 89 de                     	movq	%rbx, %rsi
      3a: 4c 89 fa                     	movq	%r15, %rdx
      3d: e8 8e 00 00 00               	callq	0xd0 <forge_add_6>
      42: 4d 8b 7e 18                  	movq	0x18(%r14), %r15
      46: c5 fa 10 05 00 00 00 00      	vmovss	(%rip), %xmm0           # 0x4e <forge_kernel+0x4e>
      4e: 4c 89 ff                     	movq	%r15, %rdi
      51: 48 89 de                     	movq	%rbx, %rsi
      54: 4c 89 e2                     	movq	%r12, %rdx
      57: e8 c4 00 00 00               	callq	0x120 <forge_multiply_9>
      5c: 49 8b 3e                     	movq	(%r14), %rdi
      5f: 48 89 de                     	movq	%rbx, %rsi
      62: 4c 89 fa                     	movq	%r15, %rdx
      65: 48 83 c4 08                  	addq	$0x8, %rsp
      69: 5b                           	popq	%rbx
      6a: 41 5c                        	popq	%r12
      6c: 41 5e                        	popq	%r14
      6e: 41 5f                        	popq	%r15
      70: e9 fb 00 00 00               	jmp	0x170 <forge_sqrt_10>
      75: 66 66 2e 0f 1f 84 00 00 00 00 00     	nopw	%cs:(%rax,%rax)

0000000000000080 <forge_multiply_3>:
      80: 48 85 f6                     	testq	%rsi, %rsi
      83: 74 42                        	je	0xc7 <forge_multiply_3+0x47>
      85: 48 83 fe 08                  	cmpq	$0x8, %rsi
      89: 73 04                        	jae	0x8f <forge_multiply_3+0xf>
      8b: 31 c0                        	xorl	%eax, %eax
      8d: eb 26                        	jmp	0xb5 <forge_multiply_3+0x35>
      8f: 48 89 f0                     	movq	%rsi, %rax
      92: 48 83 e0 f8                  	andq	$-0x8, %rax
      96: c4 e2 7d 18 c8               	vbroadcastss	%xmm0, %ymm1
      9b: 31 c9                        	xorl	%ecx, %ecx
      9d: 0f 1f 00                     	nopl	(%rax)
      a0: c5 f4 59 14 8a               	vmulps	(%rdx,%rcx,4), %ymm1, %ymm2
      a5: c5 fc 11 14 8f               	vmovups	%ymm2, (%rdi,%rcx,4)
      aa: 48 83 c1 08                  	addq	$0x8, %rcx
      ae: 48 39 c8                     	cmpq	%rcx, %rax
      b1: 75 ed                        	jne	0xa0 <forge_multiply_3+0x20>
      b3: eb 0d                        	jmp	0xc2 <forge_multiply_3+0x42>
      b5: c5 fa 59 0c 82               	vmulss	(%rdx,%rax,4), %xmm0, %xmm1
      ba: c5 fa 11 0c 87               	vmovss	%xmm1, (%rdi,%rax,4)
      bf: 48 ff c0                     	incq	%rax
      c2: 48 39 c6                     	cmpq	%rax, %rsi
      c5: 75 ee                        	jne	0xb5 <forge_multiply_3+0x35>
      c7: c5 f8 77                     	vzeroupper
      ca: c3                           	retq
      cb: 0f 1f 44 00 00               	nopl	(%rax,%rax)

00000000000000d0 <forge_add_6>:
      d0: 48 85 f6                     	testq	%rsi, %rsi
      d3: 74 42                        	je	0x117 <forge_add_6+0x47>
      d5: 48 83 fe 08                  	cmpq	$0x8, %rsi
      d9: 73 04                        	jae	0xdf <forge_add_6+0xf>
      db: 31 c0                        	xorl	%eax, %eax
      dd: eb 26                        	jmp	0x105 <forge_add_6+0x35>
      df: 48 89 f0                     	movq	%rsi, %rax
      e2: 48 83 e0 f8                  	andq	$-0x8, %rax
      e6: c4 e2 7d 18 c8               	vbroadcastss	%xmm0, %ymm1
      eb: 31 c9                        	xorl	%ecx, %ecx
      ed: 0f 1f 00                     	nopl	(%rax)
      f0: c5 f4 58 14 8a               	vaddps	(%rdx,%rcx,4), %ymm1, %ymm2
      f5: c5 fc 11 14 8f               	vmovups	%ymm2, (%rdi,%rcx,4)
      fa: 48 83 c1 08                  	addq	$0x8, %rcx
      fe: 48 39 c8                     	cmpq	%rcx, %rax
     101: 75 ed                        	jne	0xf0 <forge_add_6+0x20>
     103: eb 0d                        	jmp	0x112 <forge_add_6+0x42>
     105: c5 fa 58 0c 82               	vaddss	(%rdx,%rax,4), %xmm0, %xmm1
     10a: c5 fa 11 0c 87               	vmovss	%xmm1, (%rdi,%rax,4)
     10f: 48 ff c0                     	incq	%rax
     112: 48 39 c6                     	cmpq	%rax, %rsi
     115: 75 ee                        	jne	0x105 <forge_add_6+0x35>
     117: c5 f8 77                     	vzeroupper
     11a: c3                           	retq
     11b: 0f 1f 44 00 00               	nopl	(%rax,%rax)

0000000000000120 <forge_multiply_9>:
     120: 48 85 f6                     	testq	%rsi, %rsi
     123: 74 42                        	je	0x167 <forge_multiply_9+0x47>
     125: 48 83 fe 08                  	cmpq	$0x8, %rsi
     129: 73 04                        	jae	0x12f <forge_multiply_9+0xf>
     12b: 31 c0                        	xorl	%eax, %eax
     12d: eb 26                        	jmp	0x155 <forge_multiply_9+0x35>
     12f: 48 89 f0                     	movq	%rsi, %rax
     132: 48 83 e0 f8                  	andq	$-0x8, %rax
     136: c4 e2 7d 18 c8               	vbroadcastss	%xmm0, %ymm1
     13b: 31 c9                        	xorl	%ecx, %ecx
     13d: 0f 1f 00                     	nopl	(%rax)
     140: c5 f4 59 14 8a               	vmulps	(%rdx,%rcx,4), %ymm1, %ymm2
     145: c5 fc 11 14 8f               	vmovups	%ymm2, (%rdi,%rcx,4)
     14a: 48 83 c1 08                  	addq	$0x8, %rcx
     14e: 48 39 c8                     	cmpq	%rcx, %rax
     151: 75 ed                        	jne	0x140 <forge_multiply_9+0x20>
     153: eb 0d                        	jmp	0x162 <forge_multiply_9+0x42>
     155: c5 fa 59 0c 82               	vmulss	(%rdx,%rax,4), %xmm0, %xmm1
     15a: c5 fa 11 0c 87               	vmovss	%xmm1, (%rdi,%rax,4)
     15f: 48 ff c0                     	incq	%rax
     162: 48 39 c6                     	cmpq	%rax, %rsi
     165: 75 ee                        	jne	0x155 <forge_multiply_9+0x35>
     167: c5 f8 77                     	vzeroupper
     16a: c3                           	retq
     16b: 0f 1f 44 00 00               	nopl	(%rax,%rax)

0000000000000170 <forge_sqrt_10>:
     170: 48 85 f6                     	testq	%rsi, %rsi
     173: 74 46                        	je	0x1bb <forge_sqrt_10+0x4b>
     175: 48 83 fe 08                  	cmpq	$0x8, %rsi
     179: 73 04                        	jae	0x17f <forge_sqrt_10+0xf>
     17b: 31 c0                        	xorl	%eax, %eax
     17d: eb 26                        	jmp	0x1a5 <forge_sqrt_10+0x35>
     17f: 48 89 f0                     	movq	%rsi, %rax
     182: 48 83 e0 f8                  	andq	$-0x8, %rax
     186: 31 c9                        	xorl	%ecx, %ecx
     188: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)
     190: c5 fc 51 04 8a               	vsqrtps	(%rdx,%rcx,4), %ymm0
     195: c5 fc 11 04 8f               	vmovups	%ymm0, (%rdi,%rcx,4)
     19a: 48 83 c1 08                  	addq	$0x8, %rcx
     19e: 48 39 c8                     	cmpq	%rcx, %rax
     1a1: 75 ed                        	jne	0x190 <forge_sqrt_10+0x20>
     1a3: eb 11                        	jmp	0x1b6 <forge_sqrt_10+0x46>
     1a5: c5 fa 10 04 82               	vmovss	(%rdx,%rax,4), %xmm0
     1aa: c5 fa 51 c0                  	vsqrtss	%xmm0, %xmm0, %xmm0
     1ae: c5 fa 11 04 87               	vmovss	%xmm0, (%rdi,%rax,4)
     1b3: 48 ff c0                     	incq	%rax
     1b6: 48 39 c6                     	cmpq	%rax, %rsi
     1b9: 75 ea                        	jne	0x1a5 <forge_sqrt_10+0x35>
     1bb: c5 f8 77                     	vzeroupper
     1be: c3                           	retq
