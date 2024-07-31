from typing import Callable, cast

from langchain_core.runnables import RunnableLambda

# Define the lambdas with type annotations
add_one: Callable[[int], int] = lambda x: x + 1
multiply_by_two: Callable[[int], int] = lambda x: x * 2

# Create the sequence
# sequence = RunnableLambda(add_one) | RunnableLambda(multiply_by_two)
sequence = RunnableLambda(cast(Callable[[int], int], lambda x: x + 1)) | RunnableLambda(
    cast(Callable[[int], int], lambda x: x * 2)
)
res = sequence.invoke(1)
print(res)

# sequence = RunnableLambda(lambda x: x + 1) | {
#     "mul_2": RunnableLambda(lambda x: x * 2),
#     "mul_5": RunnableLambda(lambda x: x * 5),
# }
sequence = RunnableLambda(cast(Callable[[int], int], lambda x: x + 1)) | {
    "mul_2": RunnableLambda(cast(Callable[[int], int], lambda x: x * 2)),
    "mul_5": RunnableLambda(cast(Callable[[int], int], lambda x: x * 5)),
}
res = sequence.invoke(1)  # {'mul_2': 4, 'mul_5': 10}
print(res)

import random

# def add_one(x: int) -> int:
#     return x + 1


def buggy_double(y: int) -> int:
    """Buggy code that will fail 70% of the time"""
    if random.random() > 0.3:
        print("This code failed, and will probably be retried!")  # noqa: T201
        raise ValueError("Triggered buggy code")
    return y * 2


sequence = RunnableLambda(add_one) | RunnableLambda(
    buggy_double
).with_retry(  # Retry on failure
    stop_after_attempt=10, wait_exponential_jitter=False
)

print(sequence.input_schema.schema())  # Show inferred input schema
print(sequence.output_schema.schema())  # Show inferred output schema
print(sequence.invoke(2))  # invoke the sequence (note the retry above!!)
