~~1. 链接不到库~~
2.
```
        Constructing wrapper function "dqrinc"...
getarrdims:warning: assumed shape array, using 0 instead of '*'
getarrdims:warning: assumed shape array, using 0 instead of '*'
getarrdims:warning: assumed shape array, using 0 instead of '*'
getarrdims:warning: assumed shape array, using 0 instead of '*'
          dqrinc(m,n,k,q,r,j,x,w,[ldq,ldr])
    Generating possibly empty wrappers"
    Maybe empty "qrupdate-f2pywrappers.f"
```
assumed shape array的问题：数组的大小可能不是在编译时确定的，而是在运行时确定