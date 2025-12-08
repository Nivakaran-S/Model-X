# Debug path calculation
from pathlib import Path

# Simulate the path from data_transformation.py
file_path = Path(r"C:\Users\LENOVO\Desktop\Roger-Ultimate\models\anomaly-detection\src\components\data_transformation.py")

print("File:", file_path)
print()
print("1 up (.parent):", file_path.parent)  # components
print("2 up:", file_path.parent.parent)      # src
print("3 up:", file_path.parent.parent.parent)  # anomaly-detection
print("4 up:", file_path.parent.parent.parent.parent)  # models
print("5 up:", file_path.parent.parent.parent.parent.parent)  # Roger-Ultimate (CORRECT!)
print()

main_project = file_path.parent.parent.parent.parent.parent
print("Main project root:", main_project)
print("Should be:", r"C:\Users\LENOVO\Desktop\Roger-Ultimate")
print("Match:", str(main_project) == r"C:\Users\LENOVO\Desktop\Roger-Ultimate")

# Check if src/graphs exists
src_graphs = main_project / "src" / "graphs"
print()
print("src/graphs path:", src_graphs)
print("Exists:", src_graphs.exists())

# Check vectorizationAgentGraph
vec_graph = src_graphs / "vectorizationAgentGraph.py"
print("vectorizationAgentGraph.py:", vec_graph)
print("Exists:", vec_graph.exists())
