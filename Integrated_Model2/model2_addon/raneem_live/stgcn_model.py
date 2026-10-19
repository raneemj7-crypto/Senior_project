import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO 17-keypoint skeleton connections
COCO_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def build_adjacency(num_joints=17):
    adjacency = torch.eye(num_joints)

    for i, j in COCO_EDGES:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = degree.pow(-0.5)

    normalized = (
        degree_inv_sqrt[:, None]
        * adjacency
        * degree_inv_sqrt[None, :]
    )

    return normalized


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.spatial = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(5, 1),
            padding=(2, 0)
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, adjacency):
        residual = self.residual(x)

        # Exchange information between connected joints
        x = torch.einsum("nctv,vw->nctw", x, adjacency)

        x = self.spatial(x)
        x = self.temporal(x)
        x = self.bn(x)

        return F.relu(x + residual)


class STGCNPlusPlus(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.register_buffer(
            "adjacency",
            build_adjacency()
        )

        self.block1 = STGCNBlock(3, 32)
        self.block2 = STGCNBlock(32, 64)
        self.block3 = STGCNBlock(64, 128)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (N, 3, 30, 17, 1)
        x = x.squeeze(-1)

        x = self.block1(x, self.adjacency)
        x = self.block2(x, self.adjacency)
        x = self.block3(x, self.adjacency)

        x = x.mean(dim=(2, 3))
        x = self.dropout(x)

        return self.classifier(x)


if __name__ == "__main__":
    model = STGCNPlusPlus()

    dummy_input = torch.randn(2, 3, 30, 17, 1)

    output = model(dummy_input)

    print("Model initialized successfully!")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters())
    )