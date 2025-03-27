# Action-Slot Replication on ROAD++ dataset

This project replicates the core ideas of the [Action-Slot model](https://github.com/HCIS-Lab/Action-slot) using the [ROAD++ Dataset](https://github.com/salmank255/Road-waymo-dataset). It aims to detect and classify complex road events from autonomous driving videos, using both global (ego-view) and local (object tube) features.

Objective

To detect high-level **road situations** involving:
- Agent (e.g., Car, Pedestrian, Cyclist)
- Action (e.g., Moving, Turning, Stopping)
-  Semantic Location (e.g., Left Lane, Crosswalk)

Each event is encoded as a **triplet**:

```python
[Agent Type, Action, Location]
