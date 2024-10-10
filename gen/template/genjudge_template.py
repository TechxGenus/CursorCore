GENJUDGE_SYSTEM = """You are tasked with assisting a programmer by maintaining a record of the programming process, including potential future changes. Your role is to discern which changes the programmer desires you to propose proactively. These should align with their actual intentions and be helpful. To determine which changes align with a programmer's intentions, consider the following principles:

1. **Understand the Context**: Assess the overall goal of the programming project. Ensure that any proposed change aligns with the project's objectives and the programmer's current focus.

2. **Maintain Clear Communication**: Before proposing changes, ensure that your suggestions are clear and concise. This helps the programmer quickly understand the potential impact of each change.

3. **Prioritize Stability**: Avoid proposing changes that could introduce instability or significant complexity unless there is a clear benefit. Stability is often more valued than optimization in the early stages of development.

4. **Respect the Programmer's Preferences**: Pay attention to the programmer's coding style and preferences. Propose changes that enhance their style rather than contradict it.

5. **Incremental Improvements**: Suggest changes that offer incremental improvements rather than drastic overhauls, unless specifically requested. This approach is less disruptive and easier for the programmer to integrate.

6. **Consider Long-Term Maintenance**: Propose changes that improve code maintainability and readability. This includes refactoring for clarity, reducing redundancy, and enhancing documentation.

7. **Balance Proactivity and Reactivity**: Be proactive in suggesting improvements that are likely to be universally beneficial (e.g., bug fixes, performance enhancements). However, be reactive, not proactive, in areas where the programmer's specific intentions are unclear or where personal preference plays a significant role.

For each potential change, return `True` if suggesting this change would be beneficial to the programmer, return `False` if the change does not align with the programmer's intentions or if they do not want you to predict this change. Give your decision after analyzing each change. Provide your response in the following format:

```
**Analysis of change 1:**

Your analysis here.

**Decision:** `True` or `False`

**Analysis of change 2:**

Your analysis here.

**Decision:** `True` or `False`

...
```"""

GENJUDGE_RECORD_TYPE = {
    "current": """Current code:
{current}
""",
    "history": """Revised code changes:
{history}
"""
}

GENJUDGE_PROMPT_INPUT = """{record}
{change}"""

GENJUDGE_PROMPT_INPUT_RECORD = """**record:**
{record}"""

GENJUDGE_PROMPT_INPUT_CHANGE = """**change {num}:**
{change}"""

GENJUDGE_PROMPT_OUTPUT = """{judgement}"""

GENJUDGE_FEWSHOT = [
    {
        "record": [
            {
                "type": "history",
                "history": """```diff\n@@ -1 +1,5 @@\n+/**\n+ * magical invsqrt function from Quake III code\n+ * see: http://www.codemaestro.com/reviews/9\n+ */\n \n```"""
            },
            {
                "type": "current",
                "current": """```c\n1 /**\n2  * magical invsqrt function from Quake III code\n3  * see: http://www.codemaestro.com/reviews/9\n4  */\n5 \n```"""
            }
        ],
        "change": [
            """```diff\n@@ -3,3 +3,19 @@\n  * see: http://www.codemaestro.com/reviews/9\n  */\n \n+float InvSqrt(float x)\n+{\n+\tfloat xhalf = 0.5f*x;\n+\tint i = *(int*)&x;\n+\ti = 0x5f3759df - (i>>1);\n+\tx = *(float*)&i;\n+\tx = x*(1.5f-xhalf*x*x);\n+\treturn x;\n+}\n+\n+int main(void) {\n+\tint result = InvSqrt(0.00056);\n+\tprintf("Result: %d (should be 42)", result);\n+\treturn result != 42;\n+}\n+\n```"""
        ]
    },
    {
        "judgement": """**Analysis of change 1:**\n\nThe change introduces the `InvSqrt` function implementation and a basic test case within the `main` function. This change is appropriate and beneficial. The primary goal is to implement and test the `InvSqrt` function, a well-known algorithm from the Quake III code. Adding the function and a basic test case aligns directly with this goal. The implementation is clear and concise, following common C coding practices. The function's logic is straightforward and well-documented, making it easy to understand. Overall, the proposed change is a logical next step in developing the `InvSqrt` function and ensuring it works correctly.\n\n**Decision:** `True`"""
    },
    {
        "record": [
            {
                "type": "history",
                "history": """```diff\n@@ -1 +1,21 @@\n \n+package com.google.gwtjsonrpc.client;\n+\n+public class VoidResult_JsonSerializer extends JsonSerializer<VoidResult> {\n+  public static final VoidResult_JsonSerializer INSTANCE =\n+      new VoidResult_JsonSerializer();\n+\n+  private VoidResult_JsonSerializer() {\n+  }\n+\n+  @Override\n+  public void printJson(final StringBuffer sb, final VoidResult o) {\n+    sb.append("{}");\n+  }\n+\n+  @Override\n+  public VoidResult fromJson(final Object o) {\n+    return VoidResult.INSTANCE;\n+  }\n+}\n+\n```"""
            },
            {
                "type": "current",
                "current": """```java\n1 \n2 \n3 package com.google.gwtjsonrpc.client;\n4 \n5 public class VoidResult_JsonSerializer extends JsonSerializer<VoidResult> {\n6   public static final VoidResult_JsonSerializer INSTANCE =\n7       new VoidResult_JsonSerializer();\n8 \n9   private VoidResult_JsonSerializer() {\n10   }\n11 \n12   @Override\n13   public void printJson(final StringBuffer sb, final VoidResult o) {\n14     sb.append("{}");\n15   }\n16 \n17   @Override\n18   public VoidResult fromJson(final Object o) {\n19     return VoidResult.INSTANCE;\n20   }\n21 }\n22 \n```"""
            }
        ],
        "change": [
            """```diff@@ -1,3 +1,16 @@\n+// Copyright 2008 Google Inc.\n+//\n+// Licensed under the Apache License, Version 2.0 (the "License");\n+// you may not use this file except in compliance with the License.\n+// You may obtain a copy of the License at\n+//\n+// http://www.apache.org/licenses/LICENSE-2.0\n+//\n+// Unless required by applicable law or agreed to in writing, software\n+// distributed under the License is distributed on an "AS IS" BASIS,\n+// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n+// See the License for the specific language governing permissions and\n+// limitations under the License.\n \n package com.google.gwtjsonrpc.client;\n \n```"""
        ]
    },
    {
        "judgement": """**Analysis of change 1:**\n\nWithout enough information to confirm the use of the Apache License, predicting this change may not align with the programmer's intentions or the project's actual licensing requirements.\n\n**Decision:** `False`"""
    }
]
