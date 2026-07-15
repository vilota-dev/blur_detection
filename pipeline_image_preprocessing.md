# Complete Model & Training Pipeline (SAM3 + DINOv3 + Classifier)

This diagram visualizes the complete system architecture, data flow, and training/filtering processes extracted from the system documentation.

```mermaid
graph TD
    %% Styles
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef sam3 fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef dino fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef classifier fill:#ede7f6,stroke:#673ab7,stroke-width:2px;
    classDef training fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;

    %% 1. Input & SAM3 Section
    A[factory original image] --> B1[hard code crop image <br> according to the image number]
    B1 --> B2[segment two building]
    B2 --> B3[hard code crop again to <br> a smaller region]
    class A input;
    class B1,B2,B3 sam3;

    %% 2. Cropped Image Paths
    B3 -->|cropped image| C[cropped image]
    class C input;

    %% Path 1: Laplacian
    C --> D1[vertical laplacian operator <br> to detect horizontal line]
    D1 --> D2[Laplacian tensor <br> let classifier see raw edge info]
    class D1,D2 process;

    %% Path 2: Sobel Edge
    C --> E1[sobel edge detection]
    E1 --> E2[masked patches <br> provide edge region focus]
    class E1,E2 process;

    %% Path 3: Full Image
    C --> F1[full image <br> provide global context]
    class F1 process;

    %% 3. DINOv3 Embeddings
    E2 --> G[DINOv3 <br> dinov3_vith16plus]
    F1 --> G
    G --> H[concatenated <br> visual embedding]
    class G dino;
    class H process;

    %% 4. Concatenation
    D2 --> I[concat]
    H --> I
    I --> J[input of classifier]
    class I,J process;

    %% 5. Classification Network
    J --> K1[Linear -> Batch Norm -> ReLU]
    K1 --> K2_split{Split}
    
    %% Main Path
    K2_split --> L1[Linear -> Batch Norm -> ReLU]
    L1 --> L2[Dropout]
    L2 --> L3[Linear -> Batch Norm]
    
    %% Residual Path
    K2_split --> M1[Linear -> Batch Norm <br> Residual to preserve edge info]
    
    %% Merge
    L3 --> N[Add / Sum]
    M1 --> N
    N --> O1[ReLU]
    O1 --> O2[Linear -> Batch Norm -> ReLU]
    O2 --> O3[Dropout]
    O3 --> O4[Linear]
    class K1,L1,L2,L3,M1,N,O1,O2,O3,O4 classifier;

    %% 6. Training & Annotation Workflow
    O4 --> P[classification output]
    class P process;

    %% Ground Truth
    T1[CSV human annotation <br> ground truth] --> T2[data cleaning <br> drop unused column, fill NAN, relabel to 4 classes]
    T2 --> T3[clean CSV human annotation <br> ground truth]
    class T1,T2,T3 training;

    %% Loss & Backprop
    T3 --> U[CrossEntropyLoss + AdamW]
    P --> U
    U -->|Backpropagation| K1
    class U training;

    %% Filtering
    P --> V1[CSV prediction output]
    V1 --> V2[hardcode filter data <br> check again by classification result and confidence]
    V2 --> V3[human check filtered cases]
    V3 --> V4[flc]
    class V1,V2,V3,V4 training;
```
