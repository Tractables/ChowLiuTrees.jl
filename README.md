# ChowLiuTrees.jl

This package provides functionalities for learning Chow-Liu Trees from data. It is part of the [Juice package](https://github.com/Juice-jl) (Julia Circuit Empanada).

## Installation
You can enter Pkg mode by hitting ], and then
````
(v1.7) pkg> add ChowLiuTrees
````

## Example
See under folder `example` for example usage.

## Testing

To make sure everything is working correctly, you can run our test suite as follows. The first time you run the tests will trigger a few slow downloads of various test resources.

```bash
julia --color=yes -e 'using Pkg; Pkg.test("ChowLiuTrees")'
```
