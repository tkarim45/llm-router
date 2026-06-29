"""A labeled query set with ground-truth difficulty.

`difficulty == "hard"` means a small/cheap model gets it wrong and you need the large model;
`"easy"` means the small model is sufficient. This label is what makes routing *measurable*:
the right router sends only the hard queries to the expensive model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Query:
    prompt: str
    difficulty: str   # "easy" | "hard"


# Easy: short factual / formatting / greeting — a small model handles these fine.
_EASY = [
    "What is the capital of France?",
    "Translate 'good morning' to Spanish.",
    "Convert 10 kilometers to miles.",
    "What day comes after Tuesday?",
    "Spell the word 'necessary'.",
    "What is 7 times 8?",
    "Give a synonym for 'happy'.",
    "What color do you get mixing blue and yellow?",
    "How many days are in September?",
    "Write 'hello' in uppercase.",
    "What is the chemical symbol for water?",
    "Round 3.14159 to two decimals.",
    "Name the largest planet in the solar system.",
    "What is the plural of 'mouse'?",
    "Define the word 'rapid'.",
    "What is 100 divided by 4?",
    "Which ocean is the largest?",
    "Capitalize the first letter of 'apple'.",
    "What is the freezing point of water in Celsius?",
    "Give the past tense of 'run'.",
    "How many sides does a hexagon have?",
    "What is the opposite of 'up'?",
    "Translate 'thank you' to French.",
    "What is 15 plus 27?",
    "Name a primary color.",
    "What is the square root of 81?",
    "Abbreviate 'Doctor'.",
    "What sound does a cat make?",
    "What is the first month of the year?",
    "Convert 1 hour to minutes.",
]

# Hard: multi-step reasoning, code, ambiguity, math word problems — small model fails.
_HARD = [
    "A train leaves at 2pm going 60mph and another at 3pm going 80mph from the same station in the same direction; at what time does the second catch the first?",
    "Write a Python function that returns the longest palindromic substring of a string.",
    "Prove that the sum of the first n odd numbers equals n squared.",
    "If all Bloops are Razzies and some Razzies are Lazzies, can we conclude some Bloops are Lazzies? Explain.",
    "Refactor a recursive Fibonacci into an O(n) iterative version and explain the complexity.",
    "A store offers 20% off then an additional 15% off the discounted price; what single percentage discount is that equivalent to?",
    "Explain why the halting problem is undecidable in two paragraphs.",
    "Given a list of meetings with start/end times, write code to find the minimum number of rooms required.",
    "Three people check into a hotel for $30, the clerk refunds $5 via a bellhop who keeps $2; where did the missing dollar go? Resolve the paradox.",
    "Derive the closed-form solution for linear regression and state its assumptions.",
    "Design a rate limiter using the token-bucket algorithm and describe its edge cases.",
    "If a bat and ball cost $1.10 together and the bat costs $1 more than the ball, how much is the ball?",
    "Write SQL to find the second-highest salary without using LIMIT or TOP.",
    "Explain the CAP theorem and give a concrete system that chooses AP over CP and why.",
    "A farmer must cross a river with a wolf, a goat, and a cabbage using a boat that holds one item; give the full sequence of crossings.",
    "Implement binary search and explain why it fails on an unsorted array.",
    "What is the expected number of coin flips to get two heads in a row?",
    "Given dependencies between tasks, write code to produce a valid topological order or detect a cycle.",
    "Explain the difference between process and thread and when context-switching cost dominates.",
    "Solve: 3x + 2y = 12 and x - y = 1 for x and y, showing each step.",
    "Design an LRU cache with O(1) get and put and justify the data structures.",
    "If the probability of rain is 0.3 each day independently, what is the chance of at least one rainy day in a week?",
    "Explain why floating point 0.1 + 0.2 != 0.3 and how to compare floats safely.",
    "Write a function to detect a cycle in a linked list using O(1) extra space.",
    "Two trains 300 miles apart approach each other at 50 and 70 mph; a bird flies between them at 100 mph until they meet — how far does the bird travel?",
    "Explain eventual consistency and how vector clocks resolve conflicting writes.",
    "Given a binary tree, write code to find its maximum depth and analyze the time complexity.",
    "A 5-liter and a 3-liter jug, measure exactly 4 liters; give the steps.",
    "Explain why quicksort is O(n^2) worst case and how randomized pivots fix it.",
    "Compute the determinant of a 3x3 matrix symbolically and explain what a zero determinant means.",
]


def dataset() -> list[Query]:
    return [Query(p, "easy") for p in _EASY] + [Query(p, "hard") for p in _HARD]
