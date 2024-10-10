GENCHAT_SYSTEM = """You are a programming assistant. The following content includes information related to your programming assistance, which may contain the record of the programming process, the current code, the user instruction, and your predicted modifications. Please provide the chat conversation for making the prediction. This may include analyzing the past programming process, speculating on the user's intent, and explaining the planning and ideas for modifying the code. Return your chat conversation in the following format:
```
**chat:**
{chat}
```"""

GENCHAT_RECORD_TYPE = {
    "current": """Current code:
{current}
""",
    "history": """Revised code changes:
{history}
""",
    "user": """User instruction:
{user}
""",
}

GENCHAT_PROMPT_INPUT = """{record}
Predicted modifications:
{change}"""

GENCHAT_PROMPT_OUTPUT = """**chat:**
{chat}"""

GENCHAT_FEWSHOT = [
    {
        "record": [
            {
                "type": "history",
                "history": """```diff\n@@ -14,3 +14,30 @@\n         if (row == n) {\n             vector<string> board = generateBoard(queens, n);\n             solutions.push_back(board);\n+        } else {\n+            for (int i = 0; i < n; i++) {\n+                if (columns.find(i) != columns.end()) {\n+                    continue;\n+                }\n+                int diagonal1 = row - i;\n+                if (diagonals1.find(diagonal1) != diagonals1.end()) {\n+                    continue;\n+                }\n+                int diagonal2 = row + i;\n+                if (diagonals2.find(diagonal2) != diagonals2.end()) {\n+                    continue;\n+                }\n+                queens[row] = i;\n+                columns.insert(i);\n+                diagonals1.insert(diagonal1);\n+                diagonals2.insert(diagonal2);\n+                backtrack(solutions, queens, n, row + 1, columns, diagonals1, diagonals2);\n+                queens[row] = -1;\n+                columns.erase(i);\n+                diagonals1.erase(diagonal1);\n+                diagonals2.erase(diagonal2);\n+            }\n+        }\n+    }\n+\n+    vector<string> generateBoard(vector<int> &queens, int n)\n```"""
            },
            {
                "type": "history",
                "history": """```diff\n@@ -3,41 +3,3 @@\n     vector<vector<string>> solveNQueens(int n) {\n         auto solutions = vector<vector<string>>();\n         auto queens = vector<int>(n, -1);\n-        auto columns = unordered_set<int>();\n-        auto diagonals1 = unordered_set<int>();\n-        auto diagonals2 = unordered_set<int>();\n-        backtrack(solutions, queens, n, 0, columns, diagonals1, diagonals2);\n-        return solutions;\n-    }\n-\n-    void backtrack(vector<vector<string>> &solutions, vector<int> &queens, int n, int row, unordered_set<int> &columns, unordered_set<int> &diagonals1, unordered_set<int> &diagonals2) {\n-        if (row == n) {\n-            vector<string> board = generateBoard(queens, n);\n-            solutions.push_back(board);\n-        } else {\n-            for (int i = 0; i < n; i++) {\n-                if (columns.find(i) != columns.end()) {\n-                    continue;\n-                }\n-                int diagonal1 = row - i;\n-                if (diagonals1.find(diagonal1) != diagonals1.end()) {\n-                    continue;\n-                }\n-                int diagonal2 = row + i;\n-                if (diagonals2.find(diagonal2) != diagonals2.end()) {\n-                    continue;\n-                }\n-                queens[row] = i;\n-                columns.insert(i);\n-                diagonals1.insert(diagonal1);\n-                diagonals2.insert(diagonal2);\n-                backtrack(solutions, queens, n, row + 1, columns, diagonals1, diagonals2);\n-                queens[row] = -1;\n-                columns.erase(i);\n-                diagonals1.erase(diagonal1);\n-                diagonals2.erase(diagonal2);\n-            }\n-        }\n-    }\n-\n-    vector<string> generateBoard(vector<int> &queens, int n)\n```"""
            },
            {
                "type": "history",
                "history": """```diff\n@@ -3,3 +3,17 @@\n     vector<vector<string>> solveNQueens(int n) {\n         auto solutions = vector<vector<string>>();\n         auto queens = vector<int>(n, -1);\n+        solve(solutions, queens, n, 0, 0, 0, 0);\n+        return solutions;\n+    }\n+\n+    vector<string> generateBoard(vector<int> &queens, int n) {\n+        auto board = vector<string>();\n+        for (int i = 0; i < n; i++) {\n+            string row = string(n, '.');\n+            row[queens[i]] = 'Q';\n+            board.push_back(row);\n+        }\n+        return board;\n+    }\n+};\n```"""
            },
            {
                "type": "current",
                "current": """```cpp\n1 class Solution {\n2 public:\n3     vector<vector<string>> solveNQueens(int n) {\n4         auto solutions = vector<vector<string>>();\n5         auto queens = vector<int>(n, -1);\n6         solve(solutions, queens, n, 0, 0, 0, 0);\n7         return solutions;\n8     }\n9 \n10     vector<string> generateBoard(vector<int> &queens, int n) {\n11         auto board = vector<string>();\n12         for (int i = 0; i < n; i++) {\n13             string row = string(n, '.');\n14             row[queens[i]] = 'Q';\n15             board.push_back(row);\n16         }\n17         return board;\n18     }\n19 };\n```"""
            }
        ],
        "change": """```diff\n@@ -5,6 +5,23 @@\n         auto queens = vector<int>(n, -1);\n         solve(solutions, queens, n, 0, 0, 0, 0);\n         return solutions;\n+    }\n+\n+    void solve(vector<vector<string>> &solutions, vector<int> &queens, int n, int row, int columns, int diagonals1, int diagonals2) {\n+        if (row == n) {\n+            auto board = generateBoard(queens, n);\n+            solutions.push_back(board);\n+        } else {\n+            int availablePositions = ((1 << n) - 1) & (~(columns | diagonals1 | diagonals2));\n+            while (availablePositions != 0) {\n+                int position = availablePositions & (-availablePositions);\n+                availablePositions = availablePositions & (availablePositions - 1);\n+                int column = __builtin_ctz(position);\n+                queens[row] = column;\n+                solve(solutions, queens, n, row + 1, columns | position, (diagonals1 | position) >> 1, (diagonals2 | position) << 1);\n+                queens[row] = -1;\n+            }\n+        }\n     }\n \n     vector<string> generateBoard(vector<int> &queens, int n) {\n```"""
    },
    {
        "chat": """The revised code transitions from using sets to bitwise operations for solving the N-Queens problem, aiming to optimize the algorithm by reducing dynamic memory allocations and improving speed. Initially, the original code utilized sets (`unordered_set<int>`) to track columns and diagonals. The user likely seeks to enhance efficiency, prompting the replacement of these sets with integer bit masks. To achieve this, the `solve` function is adjusted to use bitwise operations for conflict checks, calculating `availablePositions` using bitwise AND and NOT operations to identify valid queen placements. The `__builtin_ctz` function finds the least significant bit set to 1, corresponding to a valid column, and the recursive `solve` call updates bit masks for columns and diagonals after placing a queen. This maintains the backtracking logic while leveraging bitwise operations for performance. The `generateBoard` function remains unchanged, as it simply formats the result. This approach ensures efficient, streamlined code for the N-Queens problem."""
    },
    {
        "record": [
            {
                "type": "current",
                "current": """```\n1 const express = require('express');\n2 const app = express();\n```"""
            },
            {
                "type": "user",
                "user": """Add two new routes to the Express.js application:\n\n1. `/upper`: This route should accept a query parameter `string` and return the uppercase version of the string. If no string is provided, it should return a 400 error with a message "Please provide a string".\n2. `/lower`: This route should accept a query parameter `string` and return the lowercase version of the string. If no string is provided, it should return a 400 error with a message "Please provide a string".\n\nImplement these routes by modifying the existing code."""
            }
        ],
        "change": """```diff\n@@ -1,2 +1,22 @@\n const express = require('express');\n const app = express();\n+\n+app.get('/upper', (req, res) => {\n+  let string = req.query.string;\n+  if (!string) {\n+    res.status(400).send('Please provide a string');\n+    return;\n+  }\n+  res.send(string.toUpperCase());\n+});\n+\n+app.get('/lower', (req, res) => {\n+  let string = req.query.string;\n+  if (!string) {\n+    res.status(400).send('Please provide a string');\n+    return;\n+  }\n+  res.send(string.toLowerCase());\n+});\n+\n+app.listen(3000, () => console.log('Server started'));\n```"""
    },
    {
        "chat": """We should add two new routes to the existing Express.js application. These routes should handle the conversion of a query parameter `string` to either uppercase or lowercase, depending on the route accessed. If the query parameter is not provided, the routes should return a 400 error with an appropriate message.\n\nTo implement this, we will:\n1. Define a route `/upper` that checks for the presence of the query parameter `string`. If it exists, we will convert it to uppercase and return the result. If it does not exist, we will return a 400 error with the message "Please provide a string".\n2. Define a route `/lower` with similar logic, but it will convert the `string` to lowercase.\n\nFinally, we will ensure that the server is listening on a specified port, in this case, port 3000."""
    }
]
