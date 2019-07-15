# infinite-monkey

We train a neural network to generate valid, interesting python expressions.

* The generator produces python source code

* The executor models the output of the Python interpreter

* The generator is trained to maximize the loss of the executor, giving it curiosity

* Errors are stripped down, making stdout more interesting to explore than stderr (caused by generating invalid code)
