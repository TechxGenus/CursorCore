GENINST_SYSTEM = """You are a programming assistant. The following content includes information related to your programming assistance, which may contain the record of the programming process, the current code, the git commit after all changes, relevant details about the problem, and your predicted modifications. Please generate an instruction for you to make the corresponding modifications, ensuring it resembles instructions typically given by a human programmer. The instruction may be detailed or concise and may or may not specify the location of the modification. Return the generated instruction in the following format:
```
**instruction:**
{instruction}
```"""

GENINST_RECORD_TYPE = {
    "current": """Current code:
{current}
""",
    "history": """Revised code changes:
{history}
""",
    "git": """Git commit message after all changes:
{git}
""",
    "problem": """Relevant details about the problem:
{problem}
""",
}

GENINST_PROMPT_INPUT = """{record}
Changes in predictions:
{change}"""

GENINST_PROMPT_OUTPUT = """**instruction:**
{instruction}"""

GENINST_FEWSHOT = [
    {
        "record": [
            {
                "type": "current",
                "current": """```python\n1 import dedupe\n2 \n3 def detect_fraud(transactions):\n4     deduper = dedupe.Dedupe()\n5     deduper.sample(transactions)\n6     deduper.train()\n7 \n8     clusters = deduper.match(transactions, threshold=0.5)\n9 \n10     return clusters\n11 \n12 transactions = [\n13     {'id': 1, 'amount': 100.0},\n14     {'id': 2, 'amount': 200.0},\n15     {'id': 3, 'amount': 150.0},\n16 ]\n17 \n18 fraud_clusters = detect_fraud(transactions)\n19 print(fraud_clusters)\n```"""
            }
        ],
        "change": """```diff\n@@ -15,5 +15,10 @@\n     {'id': 3, 'amount': 150.0},\n ]\n \n+# Replace '_sample' with 'sample'\n+deduper = dedupe.Dedupe()\n+deduper.sample = deduper._sample\n+del deduper._sample\n+\n fraud_clusters = detect_fraud(transactions)\n print(fraud_clusters)\n```"""
    },
    {
        "instruction": """Replace the internal '_sample' method with 'sample' in the dedupe object, approximately at line 17."""
    },
    {
        "record": [
            {
                "type": "history",
                "history": """```diff\n@@ -3,6 +3,10 @@\n def create_cnn_model(in_channels, config):\n     layers = []\n     conv2d = nn.Conv2d(in_channels, config, kernel_size=3, padding=1)\n-    layers += [conv2d, nn.ReLU(inplace=True)]\n+    if batch_norm:\n+        layers += [conv2d, nn.BatchNorm2d(config)]\n+    else:\n+        layers += [conv2d]\n+    layers += [nn.ReLU(inplace=True)]\n     model = nn.Sequential(*layers)\n     return model\n```"""
            },
            {
                "type": "history",
                "history": """```diff\n@@ -1,6 +1,6 @@\n import torch.nn as nn\n \n-def create_cnn_model(in_channels, config):\n+def create_cnn_model(in_channels, config, batch_norm=False):\n     layers = []\n     conv2d = nn.Conv2d(in_channels, config, kernel_size=3, padding=1)\n     if batch_norm:\n```"""
            },
            {
                "type": "current",
                "current": """```\n1 import torch.nn as nn\n2 \n3 def create_cnn_model(in_channels, config, batch_norm=False):\n4     layers = []\n5     conv2d = nn.Conv2d(in_channels, config, kernel_size=3, padding=1)\n6     if batch_norm:\n7         layers += [conv2d, nn.BatchNorm2d(config)]\n8     else:\n9         layers += [conv2d]\n10     layers += [nn.ReLU(inplace=True)]\n11     model = nn.Sequential(*layers)\n12     return model\n```"""
            }
        ],
        "change": """```diff\n@@ -1,12 +1,11 @@\n import torch.nn as nn\n \n-def create_cnn_model(in_channels, config, batch_norm=False):\n+def create_cnn_model(in_channels, config, batch_norm):\n     layers = []\n     conv2d = nn.Conv2d(in_channels, config, kernel_size=3, padding=1)\n     if batch_norm:\n-        layers += [conv2d, nn.BatchNorm2d(config)]\n+        layers += [conv2d, nn.BatchNorm2d(config), nn.ReLU(inplace=True)]\n     else:\n-        layers += [conv2d]\n-    layers += [nn.ReLU(inplace=True)]\n+        layers += [conv2d, nn.ReLU(inplace=True)]\n     model = nn.Sequential(*layers)\n     return model\n```"""
    },
    {
        "instruction": """Update the create_cnn_model function to ensure that the ReLU activation function is added immediately after the BatchNorm layer if batch_norm is enabled. Adjust the function signature to remove the default value for the batch_norm parameter. The updated code should handle the addition of the ReLU layer conditionally based on the batch_norm parameter."""
    },
    {
        "record": [
            {
                "type": "current",
                "current": """```ruby\n1 # frozen_string_literal: true\n2 module Extensions::DeferredWorkflowStatePersistence::Workflow; end\n3 module Extensions::DeferredWorkflowStatePersistence::Workflow::Adapter; end\n4 module Extensions::DeferredWorkflowStatePersistence::Workflow::Adapter::DeferredActiveRecord\n5   extend ActiveSupport::Concern\n6   included do\n7     include Workflow::Adapter::ActiveRecord\n8     include InstanceMethods\n9   end\n10 \n11   module InstanceMethods\n12     def persist_workflow_state(new_value)\n13       write_attribute(self.class.workflow_column, new_value)\n14       true\n15     end\n16   end\n17 end\n18 \n```"""
            },
            {
                "type": "git",
                "git": "Include WorkflowActiverecord in the state persistence extension."
            }
        ],
        "change": """```diff\n@@ -1,10 +1,12 @@\n # frozen_string_literal: true\n+require 'workflow_activerecord'\n+\n module Extensions::DeferredWorkflowStatePersistence::Workflow; end\n module Extensions::DeferredWorkflowStatePersistence::Workflow::Adapter; end\n module Extensions::DeferredWorkflowStatePersistence::Workflow::Adapter::DeferredActiveRecord\n   extend ActiveSupport::Concern\n   included do\n-    include Workflow::Adapter::ActiveRecord\n+    include WorkflowActiverecord::Adapter::ActiveRecord\n     include InstanceMethods\n   end\n \n```"""
    },
    {
        "instruction": """At the beginning of the file, add the statement `require 'workflow_activerecord'`; On line 7, change `include Workflow::Adapter::ActiveRecord` to `include WorkflowActiverecord::Adapter::ActiveRecord`; Ensure the final code reflects the necessary changes for including `WorkflowActiverecord` in the state persistence extension."""
    },
    {
        "record": [
            {
                "type": "history",
                "history": """```diff\n@@ -1,4 +1,5 @@\n import java.util.*;\n+import java.io.*;\n \n class Main\n {\n@@ -15,14 +16,18 @@\n                        }\n                }\n \n-               int n;\n-               Scanner sc = new Scanner(System.in);\n-               n = sc.nextInt();\n+               int ans = 0;\n+               BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));\n \n-               int ans = 0;\n-               for(Map.Entry<Integer, Integer> e : map.entrySet()){\n-                       Integer v = map.get(n-e.getKey());\n-                       ans += (v==null ? 0 : v) * e.getValue();\n+               try{\n+                       int n = Integer.parseInt(bf.readLine());\n+                       for(Map.Entry<Integer, Integer> e : map.entrySet()){\n+                               Integer v = map.get(n-e.getKey());\n+                               ans += (v==null ? 0 : v) * e.getValue();\n+                       }\n+               }\n+               catch(IOException ex){\n+                       System.out.println(ex);\n                }\n \n                System.out.println(ans);\n```"""
            },
            {
                "type": "current",
                "current": """```java\n1 import java.util.*;\n2 import java.io.*;\n3 \n4 class Main\n5 {\n6       public static void main(String[] args)\n7       {\n8               HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();\n9 \n10              for(int i=1; i<10; ++i){\n11                      for(int j=1; j<10; ++j){\n12                              if(!map.containsKey(i+j))\n13                                      map.put(i+j, 1);\n14                              else\n15                                      map.put(i+j, map.get(i+j)+1);\n16                      }\n17              }\n18 \n19              int ans = 0;\n20              BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));\n21 \n22              try{\n23                      int n = Integer.parseInt(bf.readLine());\n24                      for(Map.Entry<Integer, Integer> e : map.entrySet()){\n25                              Integer v = map.get(n-e.getKey());\n26                              ans += (v==null ? 0 : v) * e.getValue();\n27                      }\n28              }\n29              catch(IOException ex){\n30                      System.out.println(ex);\n31              }\n32 \n33              System.out.println(ans);\n34      }\n35 }\n```"""
            },
            {
                "type": "problem",
                "problem": """Problem Description:\nWrite a program which reads an integer n  and identifies the number of combinations of a, b, c and d (0 ≤ a, b, c, d ≤ 9) which meet the following equality:a + b + c + d = n\nFor example, for n = 35, we have 4 different combinations of (a, b, c, d): (8, 9, 9, 9), (9, 8, 9, 9), (9, 9, 8, 9), and (9, 9, 9, 8).\n\nInput Description:\nThe input consists of several datasets. Each dataset consists of n (1 ≤ n ≤ 50) in a line. The number of datasets is less than or equal to 50.\n\nOutput Description:\nPrint the number of combination in a line.\n\nSample Input 1:\n35\n1\n\nSample Output 1:\n4\n4"""
            }
        ],
        "change": """```diff\n@@ -7,8 +7,8 @@\n        {\n                HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();\n \n-               for(int i=1; i<10; ++i){\n-                       for(int j=1; j<10; ++j){\n+               for(int i=0; i<10; ++i){\n+                       for(int j=0; j<10; ++j){\n                                if(!map.containsKey(i+j))\n                                        map.put(i+j, 1);\n                                else\n@@ -16,20 +16,22 @@\n                        }\n                }\n \n-               int ans = 0;\n                BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));\n+               String str;\n \n                try{\n-                       int n = Integer.parseInt(bf.readLine());\n-                       for(Map.Entry<Integer, Integer> e : map.entrySet()){\n-                               Integer v = map.get(n-e.getKey());\n-                               ans += (v==null ? 0 : v) * e.getValue();\n+                       while((str = bf.readLine()) != null){\n+                               int ans = 0;\n+                               int n = Integer.parseInt(str);\n+                               for(Map.Entry<Integer, Integer> e : map.entrySet()){\n+                                       Integer v = map.get(n-e.getKey());\n+                                       ans += (v==null ? 0 : v) * e.getValue();\n+                               }\n+                               System.out.println(ans);\n                        }\n                }\n                catch(IOException ex){\n                        System.out.println(ex);\n                }\n-\n-               System.out.println(ans);\n        }\n }\n```"""
    },
    {
        "instruction": """Update the code to handle multiple datasets and calculate the number of combinations of four digits (0 ≤ a, b, c, d ≤ 9) that sum up to a given number n for each dataset. Initialize the map to store the counts of all possible sums of pairs of digits (a+b). Modify the code to read multiple inputs until EOF and for each input, calculate the number of valid combinations where the sum of two pairs equals n."""
    },
]
