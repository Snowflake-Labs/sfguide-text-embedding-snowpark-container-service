# Lodis: Local Dictionary Server

Redis is a Remote Dictionary Server. This is a Python-native dictionary server that is backed in shared memory. It's _local_ to a machine.

## Inspiration

- The hashmap is based loosely on [Numba-Fasthashmap](https://github.com/mortacious/numba-fasthashmap/tree/master) which was inspired by [C99 sc](https://github.com/tezc/sc).
- The use of `libc` semaphores for cross-process locking was taken from a Numba discourse thread on [how to use locks in nopython mode](https://numba.discourse.group/t/how-to-use-locks-in-nopython-mode/221/2)
- The queue is of my own machinations.
