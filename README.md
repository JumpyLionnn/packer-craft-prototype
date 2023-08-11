# packer-craft-prototype
a minecraft scripting language

This repo is experimental
if i were to make an actual language for that i would start from scratch with better planning and better code

## the language has 2 datatypes
1. integers - 32 bit integers
2. booleans - represented by a 32 bit integer

additionally the language also supports static string but doesnt allow any operations with them (currently only the print function supports them)

## there are a few operators
1. \+ add 2 numbers
2. \- substruct 2 sumbers
3. \* multiply 2 numbers
4. / divide 2 numbers (integer division), note: division by 0 will be treated as division by 1
5. == checks if 2 values are equal
5. \> greater than
6. < less than
7. ! boolean not
8. \- negate a number

## variables are declared with the var keyword
```
var a = 8;
```

## there are also if statements
```
if condition {
    code to run if condition results in a value greater than 0
}
else {
    code to run if the condition results in 0
}
```

## there is also a debug function to print text and numbers to the chat
```
var b = 10;
var num = b + 4;
log("the result is: ", num, " and it was created with", b);
```