# Copyright 2026 Candace Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The toy suite: small Python functions graded by hidden unit tests.

Each task gives the agent a specification with one or two examples; the hidden
tests add the edge cases the specification states but the examples do not
show. The harness never sees `tests`. EVOLVE and HELDOUT are fixed by id so a
run is reproducible; the held-out ids never enter selection.
"""

TASKS = [
    {"id": "roman_to_int", "entry": "roman_to_int",
     "prompt": "Write `roman_to_int(s: str) -> int` converting a Roman numeral "
               "(I, V, X, L, C, D, M; subtractive pairs IV, IX, XL, XC, CD, CM) to "
               "an integer. Input may be upper or lower case. Example: "
               "roman_to_int('XIV') == 14.",
     "tests": """
assert roman_to_int('XIV') == 14
assert roman_to_int('mcmxciv') == 1994
assert roman_to_int('MMMCMXCIX') == 3999
assert roman_to_int('iv') == 4
assert roman_to_int('I') == 1
"""},
    {"id": "int_to_roman", "entry": "int_to_roman",
     "prompt": "Write `int_to_roman(n: int) -> str` for 1 <= n <= 3999 using "
               "standard subtractive notation, upper case. Raise ValueError "
               "outside that range. Example: int_to_roman(9) == 'IX'.",
     "tests": """
assert int_to_roman(9) == 'IX'
assert int_to_roman(1994) == 'MCMXCIV'
assert int_to_roman(3999) == 'MMMCMXCIX'
assert int_to_roman(40) == 'XL'
for bad in (0, 4000, -3):
    try:
        int_to_roman(bad)
        raise AssertionError('no ValueError for %r' % bad)
    except ValueError:
        pass
"""},
    {"id": "rle_encode", "entry": "rle_encode",
     "prompt": "Write `rle_encode(s: str) -> str`: run-length encode as "
               "<char><count> for every run, count always written, runs are "
               "case-sensitive. The empty string encodes to ''. Example: "
               "rle_encode('aaab') == 'a3b1'.",
     "tests": """
assert rle_encode('aaab') == 'a3b1'
assert rle_encode('') == ''
assert rle_encode('aAa') == 'a1A1a1'
assert rle_encode('z' * 12) == 'z12'
assert rle_encode('112') == '1221'
"""},
    {"id": "parse_duration", "entry": "parse_duration",
     "prompt": "Write `parse_duration(s: str) -> int` returning seconds for "
               "strings made of <int><unit> parts with units d, h, m, s in that "
               "order, each at most once, e.g. '1h30m' -> 5400. Surrounding "
               "whitespace is allowed. Raise ValueError for an empty string, an "
               "unknown unit, a repeated or out-of-order unit, or a bare number.",
     "tests": """
assert parse_duration('1h30m') == 5400
assert parse_duration('45s') == 45
assert parse_duration(' 2d ') == 172800
assert parse_duration('1d1h1m1s') == 90061
for bad in ('', '10', '5x', '1m1h', '1h1h', 'h'):
    try:
        parse_duration(bad)
        raise AssertionError('no ValueError for %r' % bad)
    except ValueError:
        pass
"""},
    {"id": "merge_intervals", "entry": "merge_intervals",
     "prompt": "Write `merge_intervals(xs: list[tuple[int, int]]) -> "
               "list[tuple[int, int]]` merging overlapping or touching closed "
               "intervals ((1,2) and (2,3) merge) and returning them sorted. The "
               "input may be unsorted or empty. Example: "
               "merge_intervals([(1,3),(2,6),(8,10)]) == [(1,6),(8,10)].",
     "tests": """
assert merge_intervals([(1,3),(2,6),(8,10)]) == [(1,6),(8,10)]
assert merge_intervals([]) == []
assert merge_intervals([(5,6),(1,2),(2,3)]) == [(1,3),(5,6)]
assert merge_intervals([(1,10),(2,3)]) == [(1,10)]
"""},
    {"id": "wildcard_match", "entry": "wildcard_match",
     "prompt": "Write `wildcard_match(pattern: str, s: str) -> bool` where '?' "
               "matches exactly one character and '*' matches any sequence "
               "(including empty). The whole string must match. Example: "
               "wildcard_match('a*c', 'abbc') is True.",
     "tests": """
assert wildcard_match('a*c', 'abbc')
assert wildcard_match('*', '')
assert not wildcard_match('?', '')
assert wildcard_match('a?c*', 'abcxyz')
assert not wildcard_match('a*b', 'acbc')
assert wildcard_match('**a**', 'a')
assert wildcard_match('*' * 30 + 'b', 'a' * 40 + 'b')
"""},
    {"id": "balanced", "entry": "balanced",
     "prompt": "Write `balanced(s: str) -> bool`: True if every (), [], {} is "
               "properly nested. Characters inside single- or double-quoted "
               "string literals are ignored (a quote opens a literal that ends at "
               "the next identical quote). Example: balanced('f(\"(\")') is True.",
     "tests": """
assert balanced('f(\"(\")')
assert balanced('')
assert not balanced('(]')
assert balanced('{[()]}')
assert not balanced('((')
assert balanced(\"a['}']\")
assert not balanced(')(')
"""},
    {"id": "semver_cmp", "entry": "semver_cmp",
     "prompt": "Write `semver_cmp(a: str, b: str) -> int` returning -1, 0 or 1 "
               "comparing two semantic versions MAJOR.MINOR.PATCH with an "
               "optional -prerelease of dot-separated identifiers. Follow SemVer "
               "2.0: a prerelease sorts before the release, numeric identifiers "
               "compare numerically and sort before alphanumeric ones, and a "
               "shorter identifier list sorts first when all shared ones are "
               "equal. Example: semver_cmp('1.0.0-alpha', '1.0.0') == -1.",
     "tests": """
assert semver_cmp('1.0.0-alpha', '1.0.0') == -1
assert semver_cmp('1.10.0', '1.9.0') == 1
assert semver_cmp('1.0.0', '1.0.0') == 0
assert semver_cmp('1.0.0-alpha.1', '1.0.0-alpha') == 1
assert semver_cmp('1.0.0-alpha.2', '1.0.0-alpha.10') == -1
assert semver_cmp('1.0.0-1', '1.0.0-alpha') == -1
assert semver_cmp('1.0.0-beta', '1.0.0-alpha.9') == 1
"""},
    {"id": "word_wrap", "entry": "word_wrap",
     "prompt": "Write `word_wrap(text: str, width: int) -> list[str]`: greedy "
               "wrap on whitespace, lines at most `width` characters, words kept "
               "whole unless a word is longer than `width`, in which case it is "
               "split into width-sized chunks that are then wrapped like words. Runs of whitespace collapse. "
               "Example: word_wrap('the quick brown fox', 10) == ['the quick', "
               "'brown fox'].",
     "tests": """
assert word_wrap('the quick brown fox', 10) == ['the quick', 'brown fox']
assert word_wrap('', 5) == []
assert word_wrap('abcdefghij', 4) == ['abcd', 'efgh', 'ij']
assert word_wrap('a   b', 3) == ['a b']
assert word_wrap('aa bbbbbbb c', 3) == ['aa', 'bbb', 'bbb', 'b c']
"""},
    {"id": "spiral_order", "entry": "spiral_order",
     "prompt": "Write `spiral_order(m: list[list[int]]) -> list[int]` returning "
               "the elements of a rectangular matrix in clockwise spiral order "
               "starting at the top-left. Example: spiral_order([[1,2],[3,4]]) "
               "== [1,2,4,3].",
     "tests": """
assert spiral_order([[1,2],[3,4]]) == [1,2,4,3]
assert spiral_order([]) == []
assert spiral_order([[1,2,3]]) == [1,2,3]
assert spiral_order([[1],[2],[3]]) == [1,2,3]
assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]
"""},
    {"id": "eval_rpn", "entry": "eval_rpn",
     "prompt": "Write `eval_rpn(tokens: list[str]) -> int` evaluating integer "
               "reverse Polish notation with + - * /, where / truncates toward "
               "zero (like C). Tokens may be negative numbers such as '-3'. "
               "Example: eval_rpn(['2','1','+','3','*']) == 9.",
     "tests": """
assert eval_rpn(['2','1','+','3','*']) == 9
assert eval_rpn(['7','-2','/']) == -3
assert eval_rpn(['-7','2','/']) == -3
assert eval_rpn(['4','13','5','/','+']) == 6
assert eval_rpn(['-3']) == -3
"""},
    {"id": "flatten", "entry": "flatten",
     "prompt": "Write `flatten(x: list) -> list` flattening arbitrarily nested "
               "lists into one list. Only lists are flattened: tuples, strings "
               "and other values are kept as single elements. Example: "
               "flatten([1,[2,[3]]]) == [1,2,3].",
     "tests": """
assert flatten([1,[2,[3]]]) == [1,2,3]
assert flatten([]) == []
assert flatten([[[]]]) == []
assert flatten([(1,2),[3,'ab']]) == [(1,2),3,'ab']
deep = [1]
for _ in range(2000):
    deep = [deep]
assert flatten(deep) == [1]
"""},
    {"id": "snake_case", "entry": "snake_case",
     "prompt": "Write `snake_case(name: str) -> str` converting camelCase or "
               "PascalCase identifiers to snake_case. A run of capitals is an "
               "acronym and forms one word, and digits stay attached to the "
               "preceding word. Example: snake_case('HTTPServerError') == "
               "'http_server_error'.",
     "tests": """
assert snake_case('HTTPServerError') == 'http_server_error'
assert snake_case('camelCase') == 'camel_case'
assert snake_case('already_snake') == 'already_snake'
assert snake_case('getHTTP2Response') == 'get_http2_response'
assert snake_case('A') == 'a'
assert snake_case('parseXML') == 'parse_xml'
"""},
    {"id": "count_islands", "entry": "count_islands",
     "prompt": "Write `count_islands(grid: list[str]) -> int` counting groups of "
               "'#' cells connected horizontally or vertically (not diagonally). "
               "Rows may differ in length. Example: count_islands(['#.', '.#']) "
               "== 2.",
     "tests": """
assert count_islands(['#.', '.#']) == 2
assert count_islands([]) == 0
assert count_islands(['###', '#.#', '###']) == 1
assert count_islands(['#', '', '#']) == 2
big = ['#' * 150] * 150
assert count_islands(big) == 1
"""},
    {"id": "next_permutation", "entry": "next_permutation",
     "prompt": "Write `next_permutation(xs: list[int]) -> list[int]` returning "
               "the next lexicographically greater permutation as a NEW list "
               "(the input must not be modified); the last permutation wraps to "
               "the sorted order. Example: next_permutation([1,2,3]) == [1,3,2].",
     "tests": """
assert next_permutation([1,2,3]) == [1,3,2]
assert next_permutation([3,2,1]) == [1,2,3]
assert next_permutation([1,1,5]) == [1,5,1]
assert next_permutation([]) == []
xs = [1,3,2]
assert next_permutation(xs) == [2,1,3] and xs == [1,3,2]
"""},
    {"id": "business_days", "entry": "business_days",
     "prompt": "Write `business_days(start: str, end: str) -> int` counting "
               "Monday-Friday dates d with start <= d < end, given ISO dates "
               "'YYYY-MM-DD'. If end is before start, return the negated count "
               "for the swapped range. Example: business_days('2024-01-01', "
               "'2024-01-08') == 5.",
     "tests": """
assert business_days('2024-01-01', '2024-01-08') == 5
assert business_days('2024-01-06', '2024-01-08') == 0
assert business_days('2024-01-01', '2024-01-01') == 0
assert business_days('2024-01-08', '2024-01-01') == -5
assert business_days('2024-02-28', '2024-03-02') == 3
"""},
    {"id": "normalize_path", "entry": "normalize_path",
     "prompt": "Write `normalize_path(p: str) -> str` normalizing a POSIX path: "
               "collapse repeated slashes, drop '.', resolve '..'. For absolute "
               "paths '..' at the root stays at the root; for relative paths "
               "leading '..' components are kept. The empty result of a relative "
               "path is '.'. Example: normalize_path('/a//b/../c/.') == '/a/c'.",
     "tests": """
assert normalize_path('/a//b/../c/.') == '/a/c'
assert normalize_path('/../..') == '/'
assert normalize_path('a/../..') == '..'
assert normalize_path('') == '.'
assert normalize_path('./a/') == 'a'
assert normalize_path('../a/../../b') == '../../b'
"""},
    {"id": "top_k_words", "entry": "top_k_words",
     "prompt": "Write `top_k_words(text: str, k: int) -> list[str]` returning "
               "the k most frequent words, lower-cased, where a word is a maximal "
               "run of ASCII letters or apostrophes. Ties break alphabetically. "
               "Return fewer than k if there are fewer distinct words. Example: "
               "top_k_words('b a b', 1) == ['b'].",
     "tests": """
assert top_k_words('b a b', 1) == ['b']
assert top_k_words('The the THE cat. Cat!', 2) == ['the', 'cat']
assert top_k_words('b a', 2) == ['a', 'b']
assert top_k_words('', 3) == []
assert top_k_words(\"don't stop, don't\", 1) == [\"don't\"]
"""},
    {"id": "levenshtein", "entry": "levenshtein",
     "prompt": "Write `levenshtein(a: str, b: str) -> int`, the minimum number "
               "of single-character insertions, deletions and substitutions "
               "turning a into b. Example: levenshtein('kitten', 'sitting') == 3.",
     "tests": """
assert levenshtein('kitten', 'sitting') == 3
assert levenshtein('', 'abc') == 3
assert levenshtein('same', 'same') == 0
assert levenshtein('flaw', 'lawn') == 2
assert levenshtein('a' * 800, 'b' * 800) == 800
"""},
    {"id": "csv_line", "entry": "csv_line",
     "prompt": "Write `csv_line(line: str) -> list[str]` splitting one CSV line "
               "on commas. A field may be wrapped in double quotes; inside quotes "
               "commas are literal and a doubled quote \"\" is one quote "
               "character. Example: csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd'].",
     "tests": """
assert csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd']
assert csv_line('') == ['']
assert csv_line('a,,b') == ['a', '', 'b']
assert csv_line('\"he said \"\"hi\"\"\",x') == ['he said \"hi\"', 'x']
assert csv_line('a,') == ['a', '']
"""},
    # ---- held out ---------------------------------------------------------
    {"id": "to_base", "entry": "to_base",
     "prompt": "Write `to_base(n: int, b: int) -> str` writing integer n in base "
               "b (2 <= b <= 36) with digits 0-9 then lower-case a-z, and a "
               "leading '-' for negatives. Example: to_base(255, 16) == 'ff'.",
     "tests": """
assert to_base(255, 16) == 'ff'
assert to_base(0, 2) == '0'
assert to_base(-10, 2) == '-1010'
assert to_base(35, 36) == 'z'
assert to_base(36, 36) == '10'
"""},
    {"id": "rotate", "entry": "rotate",
     "prompt": "Write `rotate(xs: list, k: int) -> list` returning xs rotated "
               "right by k positions as a new list; k may be negative (rotate "
               "left) or larger than len(xs). Example: rotate([1,2,3], 1) == "
               "[3,1,2].",
     "tests": """
assert rotate([1,2,3], 1) == [3,1,2]
assert rotate([1,2,3], -1) == [2,3,1]
assert rotate([1,2,3], 7) == [3,1,2]
assert rotate([], 5) == []
"""},
    {"id": "valid_ipv4", "entry": "valid_ipv4",
     "prompt": "Write `valid_ipv4(s: str) -> bool`: four decimal octets 0-255 "
               "separated by dots, no leading zeros (except '0' itself), no signs "
               "or whitespace. Example: valid_ipv4('192.168.0.1') is True.",
     "tests": """
assert valid_ipv4('192.168.0.1')
assert not valid_ipv4('192.168.00.1')
assert not valid_ipv4('256.1.1.1')
assert not valid_ipv4('1.1.1')
assert not valid_ipv4('1.1.1.1 ')
assert not valid_ipv4('+1.1.1.1')
assert valid_ipv4('0.0.0.0')
"""},
    {"id": "shortest_path", "entry": "shortest_path",
     "prompt": "Write `shortest_path(edges: list[tuple[str, str, int]], src: str, "
               "dst: str) -> int` returning the least total weight from src to "
               "dst in a DIRECTED graph with non-negative weights, or -1 if dst "
               "is unreachable. src == dst costs 0. Example: "
               "shortest_path([('a','b',1),('b','c',2),('a','c',5)], 'a', 'c') == 3.",
     "tests": """
assert shortest_path([('a','b',1),('b','c',2),('a','c',5)], 'a', 'c') == 3
assert shortest_path([('a','b',1)], 'b', 'a') == -1
assert shortest_path([], 'x', 'x') == 0
assert shortest_path([('a','b',0),('b','c',0)], 'a', 'c') == 0
"""},
    {"id": "median", "entry": "median",
     "prompt": "Write `median(xs: list[float]) -> float`, the median of a "
               "non-empty list (mean of the two middle values for even length); "
               "raise ValueError for an empty list. Do not modify the input. "
               "Example: median([3,1,2]) == 2.",
     "tests": """
assert median([3,1,2]) == 2
assert median([4,1,3,2]) == 2.5
xs = [5,4,3]
median(xs)
assert xs == [5,4,3]
try:
    median([])
    raise AssertionError('no ValueError')
except ValueError:
    pass
"""},
    {"id": "dedupe_ci", "entry": "dedupe_ci",
     "prompt": "Write `dedupe_ci(words: list[str]) -> list[str]` removing "
               "case-insensitive duplicates, keeping the first spelling seen and "
               "the original order. Example: dedupe_ci(['A','b','a']) == ['A','b'].",
     "tests": """
assert dedupe_ci(['A','b','a']) == ['A','b']
assert dedupe_ci([]) == []
assert dedupe_ci(['Straße','STRASSE','strasse']) == ['Straße']
assert dedupe_ci(['x','X','y','Y','x']) == ['x','y']
"""},
    {"id": "format_money", "entry": "format_money",
     "prompt": "Write `format_money(x: float) -> str` formatting with a leading "
               "'-' for negatives, a '$', comma thousands separators and exactly "
               "two decimals rounded half away from zero. Example: "
               "format_money(1234.5) == '$1,234.50'.",
     "tests": """
assert format_money(1234.5) == '$1,234.50'
assert format_money(-0.005) == '-$0.01'
assert format_money(0) == '$0.00'
assert format_money(1000000) == '$1,000,000.00'
assert format_money(2.675) == '$2.68'
"""},
    {"id": "run_lengths", "entry": "run_lengths",
     "prompt": "Write `run_lengths(xs: list[int]) -> list[tuple[int, int]]` "
               "returning (value, count) for each maximal run of equal adjacent "
               "values. Example: run_lengths([1,1,2]) == [(1,2),(2,1)].",
     "tests": """
assert run_lengths([1,1,2]) == [(1,2),(2,1)]
assert run_lengths([]) == []
assert run_lengths([3,3,3,1,3]) == [(3,3),(1,1),(3,1)]
"""},
    {"id": "title_case", "entry": "title_case",
     "prompt": "Write `title_case(s: str) -> str` capitalizing the first letter "
               "of every word and lower-casing the rest, except that the words "
               "'a', 'an', 'the', 'of', 'and', 'in' stay lower case unless they "
               "are the first or last word. Words are separated by single spaces. "
               "Example: title_case('the lord OF the rings') == 'The Lord of the "
               "Rings'.",
     "tests": """
assert title_case('the lord OF the rings') == 'The Lord of the Rings'
assert title_case('') == ''
assert title_case('what it is made of') == 'What It Is Made Of'
assert title_case('a') == 'A'
"""},
    {"id": "chunk_sum", "entry": "chunk_sum",
     "prompt": "Write `chunk_sum(xs: list[int], size: int) -> list[int]` "
               "returning the sums of consecutive chunks of `size` elements; the "
               "last chunk may be shorter. Raise ValueError if size < 1. "
               "Example: chunk_sum([1,2,3,4,5], 2) == [3,7,5].",
     "tests": """
assert chunk_sum([1,2,3,4,5], 2) == [3,7,5]
assert chunk_sum([], 3) == []
try:
    chunk_sum([1], 0)
    raise AssertionError('no ValueError')
except ValueError:
    pass
"""},
]

HELDOUT_START = "to_base"
_ids = [t["id"] for t in TASKS]
EVOLVE = _ids[:_ids.index(HELDOUT_START)]
HELDOUT = _ids[_ids.index(HELDOUT_START):]
BY_ID = {t["id"]: t for t in TASKS}
