该目录主要是把Fortran代码转换成Python代码。

要注意python和fortran的区别，比如python的数组是从0开始的，而fortran是从1开始的。

例如`dchshx`没有返回值，这是因为这意味着传递 R 到函数 `dchshx` 中时，该函数直接在原始矩阵 R 上进行修改，而无需返回一个新的矩阵。

注：生成代码大部分都是利用了numpy、普通的循环来代替blas中的函数，请在检查的时候注意这点。