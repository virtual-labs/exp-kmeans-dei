/* Main Logic for Experiment Simulation */

/* 
 * Steps Data Configuration 
 */
const stepsData = [
  {
    id: 'import_libraries',
    title: 'Importing Libraries',
    blocks: [
      {
        code: `# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
print("Libraries Imported")`,
        output: `<div class="output-success">Libraries Imported</div>`
      }
    ]
  },
  {
    id: 'reading_data',
    title: 'Loading Dataset',
    blocks: [
      {
        code: `# Reading Mall_Customers dataset
data = pd.read_csv("Mall_Customers.csv")
print("Dataset loaded successfully")`,
        output: `<div class="output-text">Dataset loaded successfully</div>`
      }
    ]
  },
  {
    id: 'data_analysis',
    title: 'Data Analysis',
    blocks: [
      {
        code: `<div class="output-success"># Display the first 5 rows of the dataset</div>
data.head()`,
        output: `<table class="data-table">
  <thead>
    <tr>
      <th></th>
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>1</td><td>Male</td><td>19</td><td>15</td><td>39</td></tr>
    <tr><td>1</td><td>2</td><td>Male</td><td>21</td><td>15</td><td>81</td></tr>
    <tr><td>2</td><td>3</td><td>Female</td><td>20</td><td>16</td><td>6</td></tr>
    <tr><td>3</td><td>4</td><td>Female</td><td>23</td><td>16</td><td>77</td></tr>
    <tr><td>4</td><td>5</td><td>Female</td><td>31</td><td>17</td><td>40</td></tr>
  </tbody>
</table>`
      },
      {
        code: `<div class="output-success"># Display the last 5 rows of the dataset</div>
data.tail()`,
        output: `<table class="data-table">
  <thead>
    <tr>
      <th></th>
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>195</td><td>196</td><td>Female</td><td>35</td><td>120</td><td>79</td></tr>
    <tr><td>196</td><td>197</td><td>Female</td><td>45</td><td>126</td><td>28</td></tr>
    <tr><td>197</td><td>198</td><td>Male</td><td>32</td><td>126</td><td>74</td></tr>
    <tr><td>198</td><td>199</td><td>Male</td><td>32</td><td>137</td><td>18</td></tr>
    <tr><td>199</td><td>200</td><td>Male</td><td>30</td><td>137</td><td>83</td></tr>
  </tbody>
</table>`
      },
      {
        code: `<div class="output-success"># Display the summary statistics of the dataset</div>
data.describe()`,
        output: `<table class="data-table stats-table">
  <thead>
    <tr>
      <th></th>
      <th>CustomerID</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>count</td><td>200.000000</td><td>200.000000</td><td>200.000000</td><td>200.000000</td></tr>
    <tr><td>mean</td><td>100.500000</td><td>38.850000</td><td>60.560000</td><td>50.200000</td></tr>
    <tr><td>std</td><td>57.879185</td><td>13.969007</td><td>26.264721</td><td>25.823522</td></tr>
    <tr><td>min</td><td>1.000000</td><td>18.000000</td><td>15.000000</td><td>1.000000</td></tr>
    <tr><td>25%</td><td>50.750000</td><td>28.750000</td><td>41.500000</td><td>34.750000</td></tr>
    <tr><td>50%</td><td>100.500000</td><td>36.000000</td><td>61.500000</td><td>50.000000</td></tr>
    <tr><td>75%</td><td>150.250000</td><td>49.000000</td><td>78.000000</td><td>73.000000</td></tr>
    <tr><td>max</td><td>200.000000</td><td>70.000000</td><td>137.000000</td><td>99.000000</td></tr>
  </tbody>
</table>`
      },
      {
        code: `<div class="output-success"># Display the information of the dataset</div>
data.info()`,
        output: `<div class="output-text">&lt;class 'pandas.core.frame.DataFrame'&gt;</div>
<div class="output-text">RangeIndex: 200 entries, 0 to 199</div>
<div class="output-text">Data columns (total 5 columns):</div>
<table class="data-table" style="width: auto;">
  <thead>
    <tr>
      <th>#</th>
      <th>Column</th>
      <th>Non-Null Count</th>
      <th>Dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>CustomerID</td><td>200 non-null</td><td>int64</td></tr>
    <tr><td>1</td><td>Gender</td><td>200 non-null</td><td>object</td></tr>
    <tr><td>2</td><td>Age</td><td>200 non-null</td><td>int64</td></tr>
    <tr><td>3</td><td>Annual Income (k$)</td><td>200 non-null</td><td>int64</td></tr>
    <tr><td>4</td><td>Spending Score (1-100)</td><td>200 non-null</td><td>int64</td></tr>
  </tbody>
</table>
<div class="output-text" style="margin-top:5px;">dtypes: int64(4), object(1)</div>
<div class="output-text">memory usage: 7.9+ KB</div>`
      },
      {
        code: `<div class="output-success"># Check for missing values</div>
data.isnull().sum()`,
        output: `<table class="data-table" style="width: auto;">
  <thead>
    <tr>
      <th>Column</th>
      <th>Missing Values</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>CustomerID</td><td>0</td></tr>
    <tr><td>Gender</td><td>0</td></tr>
    <tr><td>Age</td><td>0</td></tr>
    <tr><td>Annual Income (k$)</td><td>0</td></tr>
    <tr><td>Spending Score (1-100)</td><td>0</td></tr>
  </tbody>
</table>
<div class="output-text" style="font-size:0.8rem; margin-top:5px;">dtype: int64</div>`
      },
      {
        code: `<div class="output-success"># Display the shape of the dataset</div>
data.shape`,
        output: `<div class="output-text">(200, 5)</div>`
      }
    ]
  },
  {
    id: 'data_preprocessing',
    title: 'Data Preprocessing',
    blocks: [
      {
        code: `<div class="output-success"># Convert the 'Gender' column from categorical values (Male/Female) to numeric labels (0/1) and preview the updated dataset</div>
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data.head()`,
        output: `<table class="data-table">
  <thead>
    <tr>
      <th></th>
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>1</td><td>0</td><td>19</td><td>15</td><td>39</td></tr>
    <tr><td>1</td><td>2</td><td>0</td><td>21</td><td>15</td><td>81</td></tr>
    <tr><td>2</td><td>3</td><td>1</td><td>20</td><td>16</td><td>6</td></tr>
    <tr><td>3</td><td>4</td><td>1</td><td>23</td><td>16</td><td>77</td></tr>
    <tr><td>4</td><td>5</td><td>1</td><td>31</td><td>17</td><td>40</td></tr>
  </tbody>
</table>`
      },
      {
        code: `<div class="output-success"># Select Annual Income and Spending Score as feature variables for clustering and display a confirmation message</div>
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print("Features selected for clustering: Annual Income and Spending Score")`,
        output: `<div class="output-text">Features selected for clustering: Annual Income and Spending Score</div>`
      },
      {
        code: `# Visualize the original data points by plotting Annual Income against Spending Score before applying K-Means clustering

plt.figure(figsize=(6,5))
plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    color='blue'
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Data Points Before K-Means Clustering")
plt.grid()
plt.show()`,
        output: `<div style="text-align:left; padding:0px;">
    <h4 style="margin:0 0 10px 0;">Data Points Before K-Means Clustering</h4>
    <div style="display:flex; justify-content:flex-start;margin-top:10px; align-items:flex-start; padding-top:0;">
        <img src="./images/dpbkc.png" style="max-width:100%; max-height:350px; margin-top:0px;" alt="Data Points">
    </div>
</div>`
      }
    ]
  },
  {
    id: 'model_training',
    title: 'Model Training',
    blocks: [
      {
        code: `<div class="output-success"># Apply StandardScaler to normalize the selected features so they have zero mean and unit variance</div>
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Feature scaling complete using StandardScaler")`,
        output: `<div class="output-text">Feature scaling complete using StandardScaler</div>`
      },
      {
        code: `<div class="output-success"># Compute WCSS for different numbers of clusters and plot the Elbow Curve to identify the optimal value of K
</div>
wcss = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid()
plt.show()`,
        output: `<div style="text-align:left; padding:0px;">
    <h4 style="margin:0 0 10px 0;">Elbow Method (k=5)</h4>
    <div style="display:flex; justify-content:flex-start;margin-top:10px; align-items:flex-start; padding-top:0;">
        <img src="./images/elbow.png" style="max-width:100%; max-height:350px; margin-top:0px;" alt="Elbow Method">
    </div>
</div>`
      },
      {
        code: `<div class="output-success"># Initialize the K-Means algorithm with 5 clusters and train it on the scaled feature data</div>
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
print("Model trained successfully")`,
        output: `<img src="./images/model_train.png" alt="Sklearn Output" style="max-width: 30%; border: 1px solid #ddd; border-radius: 4px;">
</div>
<div class="output-success">Model trained successfully</div>
<div style="margin-top: 10px;">`
      }
    ]
  },
  {
    id: 'model_evaluation',
    title: 'Model Evaluation',
    blocks: [
      {
        code: `<div class="output-success"># Assign cluster labels to the original dataset and display the first 5 rows with the new 'Cluster' column</div>
labels = kmeans.labels_
data['Cluster'] = labels
data[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head()`,
        output: `<table class="data-table">
  <thead>
    <tr>
      <th></th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>15</td><td>39</td><td>4</td></tr>
    <tr><td>1</td><td>15</td><td>81</td><td>2</td></tr>
    <tr><td>2</td><td>16</td><td>6</td><td>4</td></tr>
    <tr><td>3</td><td>16</td><td>77</td><td>2</td></tr>
    <tr><td>4</td><td>17</td><td>40</td><td>4</td></tr>
  </tbody>
</table>`
      },
      {
        code: `<div class="output-success"># Visualize the clusters by plotting Annual Income against Spending Score, with colors representing cluster labels and red 'X' markers for centroids</div>
plt.figure(figsize=(8,6))
plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=labels,
    cmap='viridis'
)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:,0],
    centers[:,1],
    c='red',
    s=200,
    marker='X'
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("K-Means Customer Segmentation")
plt.show()`,
        output: `<div style="text-align:left; padding:0px;">
    <h4 style="margin:0 0 10px 0;">K-Means Customer Segmentation</h4>
    <div style="display:flex; justify-content:flex-start;margin-top:10px; align-items:flex-start; padding-top:0;">
        <img src="./images/k-means_customer_segmentation.png" style="max-width:100%; max-height:350px; margin-top:0px;" alt="Clusters">
    </div>
</div>`
      },
      {
        code: `<div class="output-success"># Calculate and display model evaluation metrics: Silhouette Score, WCSS (Inertia), and Davies-Bouldin Index</div>
score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)

wcss = kmeans.inertia_
print("WCSS (Inertia):", wcss)

db_score = davies_bouldin_score(X_scaled, labels)
print("Davies-Bouldin Index:", db_score)`,
        output: `<div class="output-text" style="margin-bottom: 5px;">Silhouette Score: 0.5546571631111091</div>
<div class="output-text" style="margin-bottom: 5px;">WCSS (Inertia): 65.56840815571681</div>
<div class="output-text" style="margin-bottom: 5px;">Davies-Bouldin Index: 0.5722356162263352</div>`
      }
    ]
  },
  {
    id: 'model_simulation',
    title: 'Model Simulation',
    blocks: [
      {
        code: `# Interactive K-Means Simulation
Click the 'Enter Interactive Animation' button to open the full-screen simulator.
Explore how K-Means Clustering segments the customer dataset in real-time.`,
        output: `<div class="output-success">Full-Screen Simulator Ready. Click the 'Enter Interactive Animation' button to start exploring.</div>`
      }
    ]
  }
];

// --- Interactive Simulation State & Logic ---
const rawDataPoints = [
  [15, 39], [15, 81], [16, 6], [16, 77], [17, 40], [17, 76], [18, 6], [18, 94], [19, 3], [19, 72],
  [19, 14], [19, 99], [20, 15], [20, 77], [20, 13], [20, 79], [21, 35], [21, 66], [23, 29], [23, 98],
  [24, 35], [24, 73], [25, 5], [25, 73], [28, 14], [28, 82], [28, 32], [28, 61], [29, 31], [29, 87],
  [30, 4], [30, 73], [33, 4], [33, 92], [33, 14], [33, 81], [34, 17], [34, 73], [37, 26], [37, 75],
  [38, 35], [38, 92], [39, 36], [39, 61], [39, 28], [39, 65], [40, 55], [40, 47], [40, 42], [40, 42],
  [42, 52], [42, 60], [43, 54], [43, 60], [43, 45], [43, 41], [44, 50], [44, 46], [46, 51], [46, 46],
  [46, 56], [46, 55], [47, 52], [47, 59], [48, 51], [48, 59], [48, 50], [48, 48], [48, 59], [48, 47],
  [49, 55], [49, 42], [50, 49], [50, 56], [54, 47], [54, 54], [54, 53], [54, 48], [54, 52], [54, 42],
  [54, 51], [54, 55], [54, 41], [54, 44], [54, 57], [54, 46], [57, 58], [57, 55], [58, 60], [58, 46],
  [59, 55], [59, 41], [60, 49], [60, 40], [60, 42], [60, 52], [60, 47], [60, 50], [61, 42], [61, 49],
  [62, 41], [62, 48], [62, 59], [62, 55], [62, 56], [62, 42], [63, 50], [63, 46], [63, 43], [63, 48],
  [63, 52], [63, 54], [64, 42], [64, 46], [65, 48], [65, 50], [65, 43], [65, 59], [67, 43], [67, 57],
  [67, 56], [67, 40], [69, 58], [69, 91], [70, 29], [70, 77], [71, 35], [71, 95], [71, 11], [71, 75],
  [71, 9], [71, 75], [72, 34], [72, 71], [73, 5], [73, 88], [73, 7], [73, 73], [74, 10], [74, 72],
  [75, 5], [75, 93], [76, 40], [76, 87], [77, 12], [77, 97], [77, 36], [77, 74], [78, 22], [78, 90],
  [78, 17], [78, 88], [78, 20], [78, 76], [78, 16], [78, 89], [78, 1], [78, 78], [78, 1], [78, 73],
  [79, 35], [79, 83], [81, 5], [81, 93], [85, 26], [85, 75], [86, 20], [86, 95], [87, 27], [87, 63],
  [87, 13], [87, 75], [87, 10], [87, 92], [88, 13], [88, 86], [88, 15], [88, 69], [93, 14], [93, 90],
  [97, 32], [97, 86], [98, 15], [98, 88], [99, 39], [99, 97], [101, 24], [101, 68], [103, 17], [103, 85],
  [103, 23], [103, 69], [113, 8], [113, 91], [120, 16], [120, 79], [126, 28], [126, 74], [137, 18], [137, 83]
];

const simState = {
  points: [],
  centroids: [],
  k: 5,
  iteration: 0,
  converged: false,
  isAutoRunning: false,
  stepStage: 'ASSIGN',
  animationId: null,
  xMin: 0, xMax: 140,
  yMin: 0, yMax: 100,
  padding: 70,
  viewMode: 'STANDARD',
  historyStack: []
};

const CLUSTER_COLORS = [
  '#f72585', // Pink
  '#4cc9f0', // Sky Blue
  '#ffdb58', // Mustard Yellow
  '#7209b7', // Purple
  '#34d399', // Emerald Green
  '#f97316', // Orange
  '#ef4444', // Red
  '#3b82f6', // Bright Blue
  '#8b5cf6', // Violet
  '#ec4899'  // Pink
];

// State Management
let hasCompletedOnce = sessionStorage.getItem('kmeans_completed') === 'true';

let STATE = {
  stepIndex: 0,
  subStepIndex: 0,
  stepsStatus: stepsData.map(() => ({ unlocked: false, completed: false, partial: false }))
};

// Initial State: First step unlocked
STATE.stepsStatus[0].unlocked = true;

// DOM Elements
const stepsContainer = document.getElementById('stepsContainer');
const codeDisplay = document.getElementById('codeDisplay');
const outputContent = document.getElementById('outputDisplay');
const bottomPane = document.querySelector('.bottom-pane');
const runBtn = document.getElementById('runBtn');

// Initialize UI
function init() {
  renderSidebar();
  loadStep(0);
}

// Render Sidebar with Color Logic
function renderSidebar() {
  stepsContainer.innerHTML = '';

  stepsData.forEach((step, index) => {
    const status = STATE.stepsStatus[index];
    const btn = document.createElement('button');
    btn.classList.add('step-btn');

    // Label Logic
    let label = `${index + 1}. ${step.title}`;
    if (status.completed) label = `✓ ${step.title}`;
    btn.innerText = label;

    if (status.unlocked) {
      if (status.completed) {
        btn.classList.add('completed');
      } else if (status.partial) {
        btn.classList.add('in-progress');
      }

      btn.disabled = false;
      btn.style.cursor = 'pointer';

      if (index === STATE.stepIndex) {
        btn.classList.add('active');
      }

      btn.onclick = () => loadStep(index);
    } else {
      btn.classList.add('disabled');
      btn.style.color = '#888';
      btn.style.cursor = 'not-allowed';
      btn.disabled = true;
    }

    stepsContainer.appendChild(btn);
  });

  // Restart Button
  const restartBtn = document.createElement('button');
  restartBtn.classList.add('step-btn');
  restartBtn.innerText = "Restart Experiment";
  restartBtn.style.backgroundColor = "#333";
  restartBtn.style.textAlign = 'center';
  restartBtn.style.marginTop = "auto";
  restartBtn.style.color = "white";
  restartBtn.onclick = restartExperiment;
  stepsContainer.appendChild(restartBtn);

  // Download Button
  const downloadBtn = document.createElement('button');
  downloadBtn.classList.add('step-btn');
  downloadBtn.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align: middle;">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="7 10 12 15 17 10"></polyline>
      <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
    Download Experiment
  `;
  downloadBtn.style.textAlign = 'center';
  downloadBtn.style.marginTop = "10px";

  // Check if all steps are completed (or were completed before a restart)
  const allCompleted = checkAllStepsCompleted() || hasCompletedOnce;
  if (allCompleted) {
    downloadBtn.style.backgroundColor = "#f57c2a";
    downloadBtn.style.color = "white";
    downloadBtn.style.opacity = "1";
    downloadBtn.style.cursor = "pointer";
    downloadBtn.disabled = false;
    downloadBtn.onclick = downloadPDF;
  } else {
    downloadBtn.style.backgroundColor = "#f5f5f5";
    downloadBtn.style.color = "#9e9e9e";
    downloadBtn.style.opacity = "1";
    downloadBtn.style.cursor = "default";
    downloadBtn.style.border = "1px solid #e0e0e0";
    downloadBtn.disabled = false;
    downloadBtn.title = "Need to run the Experiment to download the pdf.";
    downloadBtn.onclick = function () {
      alert("Need to run the Experiment to download the pdf.");
    };
  }
  stepsContainer.appendChild(downloadBtn);
}

// Function to check if all steps are completed
function checkAllStepsCompleted() {
  return STATE.stepsStatus.every(status => status.completed);
}

function loadStep(index) {
  STATE.stepIndex = index;
  STATE.subStepIndex = 0;
  renderSidebar();
  updateUI();
}

function updateUI() {
  const step = stepsData[STATE.stepIndex];
  const block = step.blocks[STATE.subStepIndex];

  // Reset UI visibility
  const standardUI = document.getElementById('standardUI');
  const simulationPage = document.getElementById('simulationPage');
  const mainContainer = document.querySelector('.container');

  standardUI.classList.remove('hidden');
  mainContainer.classList.remove('hidden');
  simulationPage.classList.add('hidden');
  stopAutoRun(); // Ensure auto-run stops if switching back

  // Comment Header
  const commentMatch = block.code.match(/#\s*([^<\n\r]*)/);
  const codeHeaderBar = document.getElementById('codeHeaderBar');
  if (commentMatch) {
    codeHeaderBar.innerText = "# " + commentMatch[1].trim();
    codeHeaderBar.style.display = 'block';
  } else {
    codeHeaderBar.style.display = 'none';
  }

  // Code Display
  const codeWithoutTags = block.code.replace(/<[^>]*>/g, '');
  const codeWithoutComment = codeWithoutTags.replace(/#\s*.*/, '').trim();
  codeDisplay.innerHTML = highlightCode(codeWithoutComment);

  // Reset Output
  bottomPane.classList.remove('active-output');
  bottomPane.style.display = '';
  bottomPane.style.flexDirection = '';
  bottomPane.style.justifyContent = '';
  bottomPane.style.alignItems = '';

  // Special Handling for Model Simulation - LARGE BUTTON
  if (step.id === 'model_simulation') {
    outputContent.innerHTML = `
      <div class="experiment-completed-banner">
        <div style="margin-bottom:10px;"><span class="clap-emoji">\ud83d\udc4f</span> <span class="clap-emoji">\ud83d\udc4f</span> <span class="clap-emoji">\ud83d\udc4f</span></div>
        <h2>Congratulations!</h2>
        <p>You have successfully completed the K-Means Clustering experiment. You now understand how K-Means groups data into clusters and how the Elbow method helps determine the optimal number of clusters.</p>
        <button id="enterSimBtn" class="btn-enter-animation">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          Enter Interactive Animation
        </button>
      </div>
    `;
    setTimeout(() => {
      const enterBtn = document.getElementById('enterSimBtn');
      if (enterBtn) enterBtn.onclick = openSimulation;
    }, 0);

    // Enable the Download Experiment button when Model Simulation step is reached
    hasCompletedOnce = true;
    sessionStorage.setItem('kmeans_completed', 'true');
    STATE.stepsStatus.forEach(s => s.completed = true);
    renderSidebar();

    // Hide standard run button for this step to focus on the large one
    runBtn.style.display = 'none';
  } else {
    outputContent.innerHTML = '<div class="placeholder-text">Click the Run button to execute...</div>';
    runBtn.style.display = 'flex';
    runBtn.classList.remove('completed');
    runBtn.style.backgroundColor = '#F57C2A';
    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    runBtn.disabled = false;
    runBtn.onclick = runStep;
  }
}

function openSimulation() {
  const simulationPage = document.getElementById('simulationPage');
  const mainContainer = document.querySelector('.container');

  mainContainer.classList.add('hidden');
  simulationPage.classList.remove('hidden');

  // Always re-init or re-size to ensure context is correct
  setTimeout(() => {
    initInteractiveSim();
  }, 150);
}

function runStep() {
  const step = stepsData[STATE.stepIndex];
  const block = step.blocks[STATE.subStepIndex];

  outputContent.innerHTML = '<div class="loading-spinner">Running code...</div>';
  runBtn.disabled = true;

  setTimeout(() => {
    outputContent.innerHTML = block.output;
    bottomPane.classList.add('active-output');

    runBtn.classList.add('completed');
    runBtn.style.backgroundColor = '#A6CE63'; // Green matching completed steps
    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';

    STATE.stepsStatus[STATE.stepIndex].partial = true;
    renderSidebar();

    const hasNextBlock = STATE.subStepIndex < step.blocks.length - 1;

    if (hasNextBlock) {
      setTimeout(() => {
        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
        runBtn.style.backgroundColor = '#5FA8E4';
        runBtn.disabled = false;
        runBtn.onclick = nextSubStep;
      }, 500);
    } else {
      STATE.stepsStatus[STATE.stepIndex].completed = true;
      renderSidebar();

      if (STATE.stepIndex < stepsData.length - 1) {
        STATE.stepsStatus[STATE.stepIndex + 1].unlocked = true;
        renderSidebar();

        setTimeout(() => {
          runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
          runBtn.style.backgroundColor = '#5FA8E4';
          runBtn.disabled = false;
          runBtn.onclick = () => loadStep(STATE.stepIndex + 1);
        }, 500);
      } else {
        setTimeout(() => {
          runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
          runBtn.style.backgroundColor = '#72b2f7ff';
          runBtn.disabled = false;
          runBtn.onclick = () => loadStep(STATE.stepIndex);
        }, 500);
      }
    }
  }, 500);
}

function nextSubStep() {
  STATE.subStepIndex++;
  updateUI();
}

// --- Interactive Simulation Logic Implementation ---
let simCanvas, simCtx, simTooltip, simGuide;
// --- Interactive Simulation Logic Implementation (User Provided Script) ---
const state = {
  points: [],
  centroids: [],
  k: 5,
  iteration: 0,
  converged: false,
  isAutoRunning: false,
  stepStage: 'ASSIGN',
  animationId: null,
  xMin: 0, xMax: 140,
  yMin: 0, yMax: 100,
  padding: 70,
  viewMode: 'STANDARD',
  historyStack: []
};

let canvas, ctx, tooltip, guideText;
let inputs = {}, btns = {}, displays = {};

function initInteractiveSim() {
  canvas = document.getElementById('mainCanvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');
  tooltip = document.getElementById('tooltip');
  guideText = document.getElementById('guideText');

  inputs = {
    k: document.getElementById('clusterCount'),
    kVal: document.getElementById('clusterCountVal'),
    showLines: document.getElementById('showLinesToggle')
  };
  btns = {
    init: document.getElementById('btnInitCentroids'),
    prev: document.getElementById('btnPrev'),
    step: document.getElementById('btnStep'),
    auto: document.getElementById('btnAuto'),
    reset: document.getElementById('btnReset'),
    back: document.getElementById('btnBack'),
    tabs: document.querySelectorAll('.tab-btn')
  };
  displays = {
    iter: document.getElementById('iterVal'),
    legend: document.getElementById('clusterLegend')
  };

  // Back Button
  btns.back.onclick = () => {
    stopAutoRun();
    document.getElementById('simulationPage').classList.add('hidden');
    document.querySelector('.container').classList.remove('hidden');
    STATE.stepsStatus[STATE.stepIndex].completed = true;
    renderSidebar();
    updateUI();
  };

  // Controls Wiring
  inputs.k.oninput = (e) => {
    state.k = parseInt(e.target.value);
    inputs.kVal.innerText = state.k;
  };

  btns.init.onclick = () => {
    stopAutoRun();
    initData();
    initCentroids();
  };

  btns.step.onclick = stepAlgo;
  btns.auto.onclick = () => { state.isAutoRunning ? stopAutoRun() : startAutoRun(); };
  btns.reset.onclick = () => {
    stopAutoRun();
    guideText.innerHTML = "<p>Select K and click 'Initialize Centroids' to start.</p>";
    initData();
    state.centroids = [];
    state.iteration = 0;
    updateSimUI();
    draw();
  };
  btns.prev.onclick = undoStep;

  btns.tabs.forEach(tab => {
    tab.onclick = () => {
      btns.tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.viewMode = tab.dataset.view;
      draw();
    };
  });

  canvas.onmousemove = handleHover;
  inputs.showLines.onchange = draw;

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();
}

function resizeCanvas() {
  if (!canvas) return;
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  if (state.points.length > 0 || (rawDataPoints && rawDataPoints.length > 0)) {
    initData();
  }
  draw();
}

function mapDataToCanvas(income, score) {
  const p = state.padding;
  const w = canvas.width - p * 2;
  const h = canvas.height - p * 2;
  const x = p + (income / state.xMax) * w;
  const y = (canvas.height - p) - (score / state.yMax) * h;
  return { x, y, valX: income, valY: score };
}

function initData() {
  state.points = rawDataPoints.map(d => {
    const mapped = mapDataToCanvas(d[0], d[1]);
    return { x: mapped.x, y: mapped.y, valX: mapped.valX, valY: mapped.valY, clusterIndex: -1, color: '#ffffff' };
  });
}

function initCentroids() {
  state.centroids = [];
  state.iteration = 0;
  state.converged = false;
  state.stepStage = 'ASSIGN';
  state.historyStack = [];
  const shuffled = [...state.points].sort(() => 0.5 - Math.random());
  for (let i = 0; i < state.k; i++) {
    state.centroids.push({ x: shuffled[i].x, y: shuffled[i].y, color: CLUSTER_COLORS[i % CLUSTER_COLORS.length], history: [] });
  }
  guideText.innerHTML = `
    <h4>Step 0: Initialization</h4>
    <p>We placed <strong>${state.k} centroids</strong> randomly.</p>
    <p>Next: Click <strong>"Next Step"</strong> to calculate distances.</p>
  `;
  updateSimUI();
  draw();
}

function euclideanDist(p1, p2) { return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2); }

function stepAlgo() {
  if (state.converged) return;
  saveState();
  if (state.stepStage === 'ASSIGN') {
    state.points.forEach(p => {
      let minDist = Infinity, closest = -1;
      state.centroids.forEach((c, idx) => {
        const dist = euclideanDist(p, c);
        if (dist < minDist) { minDist = dist; closest = idx; }
      });
      p.clusterIndex = closest;
    });
    guideText.innerHTML = `
      <h4>Step ${state.iteration + 1}: Assignment</h4>
      <p>We calculated the distance from every customer to each centroid.</p>
      <p>Customers are now colored based on their <strong>closest centroid</strong>.</p>
      <p>Next: Move centroids to the center of their new groups.</p>
    `;
    state.stepStage = 'UPDATE';
  } else {
    let maxMove = 0;
    state.centroids.forEach((c, idx) => {
      const clusterPoints = state.points.filter(p => p.clusterIndex === idx);
      if (clusterPoints.length > 0) {
        const avgX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length;
        const avgY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length;
        maxMove = Math.max(maxMove, euclideanDist(c, { x: avgX, y: avgY }));
        c.history.push({ x: c.x, y: c.y });
        c.x = avgX; c.y = avgY;
      }
    });
    state.iteration++;
    if (maxMove < 1) {
      state.converged = true;

      // Generate results list
      let resultsHtml = `<ul style="font-size: 0.95rem; margin-top: 12px; list-style: none; padding-left: 0;">`;
      state.centroids.forEach((c, i) => {
        const valX = Math.round(((c.x - state.padding) / (canvas.width - state.padding * 2)) * state.xMax);
        const valY = Math.round(((canvas.height - c.y - state.padding) / (canvas.height - state.padding * 2)) * state.yMax);

        const colorDot = `<span style="display:inline-block; width:10px; height:10px; background:${c.color}; margin-right:8px; border-radius:50%; flex-shrink:0; border: 1px solid rgba(0,0,0,0.1);"></span>`;
        resultsHtml += `<li style="margin-bottom: 8px; color: #1e293b; display: flex; align-items: center; white-space: nowrap;">${colorDot} <strong>Cluster ${i + 1}:</strong> &nbsp; $${valX}k, Score ${valY}</li>`;
      });
      resultsHtml += `</ul>`;

      guideText.innerHTML = `
        <h4 style="color: #059669; margin-bottom: 8px; font-weight: 700;">✔ Converged!</h4>
        <p style="color: #334155; margin-bottom: 12px;">We found <strong>${state.k} customer segments</strong>.</p>
        <p style="color: #475569; font-weight: 600; margin-bottom: 5px;">Final Centroids (Crosses):</p>
        ${resultsHtml}
        <p style="margin-top: 15px; font-style: italic; color: #64748b; font-size: 0.95em;">Hover over clusters to analyze patterns!</p>
      `;
      stopAutoRun();
    } else {
      guideText.innerHTML = `
        <h4>Step ${state.iteration}: Update</h4>
        <p>Centroids moved to the center of their new groups.</p>
        <p>Next: Re-calculate distances and re-assign customers.</p>
      `;
      state.stepStage = 'ASSIGN';
    }
  }
  updateSimUI();
  draw();
}

function draw() {
  if (!ctx) return;
  // Pure White Canvas Background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (state.viewMode === 'VORONOI' && state.centroids.length > 0) drawVoronoi();
  drawAxes();
  if (inputs.showLines.checked && state.stepStage === 'UPDATE') {
    ctx.lineWidth = 1;
    state.points.forEach(p => {
      if (p.clusterIndex !== -1) {
        const c = state.centroids[p.clusterIndex];
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(c.x, c.y);
        ctx.strokeStyle = c.color; ctx.globalAlpha = 0.2; ctx.stroke(); ctx.globalAlpha = 1.0;
      }
    });
  }
  state.points.forEach(p => {
    ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = p.clusterIndex === -1 ? '#cbd5e1' : CLUSTER_COLORS[p.clusterIndex];
    ctx.fill();
    // Add subtle border to points for clarity on white background
    ctx.strokeStyle = 'rgba(0,0,0,0.1)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  });
  state.centroids.forEach(c => {
    if (c.history.length > 0) {
      ctx.beginPath(); ctx.moveTo(c.history[0].x, c.history[0].y);
      c.history.forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.lineTo(c.x, c.y); ctx.strokeStyle = c.color; ctx.globalAlpha = 0.4; ctx.stroke(); ctx.globalAlpha = 1.0;
    }
    if (state.converged) {
      const size = 12; ctx.beginPath();
      ctx.moveTo(c.x - size, c.y - size); ctx.lineTo(c.x + size, c.y + size);
      ctx.moveTo(c.x + size, c.y - size); ctx.lineTo(c.x - size, c.y + size);
      ctx.lineWidth = 5; ctx.strokeStyle = c.color; ctx.stroke();
      ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(c.x, c.y, 10, 0, Math.PI * 2);
      ctx.fillStyle = c.color; ctx.shadowBlur = 15; ctx.shadowColor = c.color; ctx.fill(); ctx.shadowBlur = 0;
      ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
    }
  });
}

function drawAxes() {
  const p = state.padding;
  ctx.strokeStyle = '#94a3b8'; // Slate 300 for soft but visible lines
  ctx.lineWidth = 1;
  ctx.font = '500 12px "Outfit"';
  ctx.fillStyle = '#475569'; // Slate 600 for text readability
  ctx.textAlign = 'center';

  // Axes
  ctx.beginPath();
  ctx.moveTo(p, p);
  ctx.lineTo(p, canvas.height - p);
  ctx.lineTo(canvas.width - p, canvas.height - p);
  ctx.stroke();

  // Grid lines and labels
  ctx.lineWidth = 0.5;
  ctx.strokeStyle = '#e2e8f0'; // Slate 200 for grid

  for (let i = 0; i <= 100; i += 20) {
    const y = (canvas.height - p) - (i / state.yMax) * (canvas.height - p * 2);
    ctx.beginPath(); ctx.moveTo(p, y); ctx.lineTo(canvas.width - p, y); ctx.stroke();
    ctx.fillText(i.toString(), p - 18, y + 4);
  }
  for (let i = 0; i <= 140; i += 20) {
    const x = p + (i / state.xMax) * (canvas.width - p * 2);
    ctx.beginPath(); ctx.moveTo(x, p); ctx.lineTo(x, canvas.height - p); ctx.stroke();
    ctx.fillText(i.toString(), x, canvas.height - p + 22);
  }

  // Titles
  ctx.font = 'bold 15px "Outfit"';
  ctx.fillStyle = '#0f172a'; // Slate 900 for titles
  ctx.fillText("Annual Income (k$)", canvas.width / 2, canvas.height - 15);

  ctx.save();
  ctx.translate(20, canvas.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Spending Score (1-100)", 0, 0);
  ctx.restore();
}

function drawVoronoi() {
  const p = state.padding, res = 4;
  for (let x = 0; x < canvas.width; x += res) {
    for (let y = 0; y < canvas.height; y += res) {
      if (x < p || x > canvas.width - p || y < p || y > canvas.height - p) continue;
      let closest = -1, minDist = Infinity;
      state.centroids.forEach((c, i) => {
        const d = (x - c.x) ** 2 + (y - c.y) ** 2;
        if (d < minDist) { minDist = d; closest = i; }
      });
      if (closest !== -1) { ctx.fillStyle = CLUSTER_COLORS[closest]; ctx.globalAlpha = 0.15; ctx.fillRect(x, y, res, res); }
    }
  }
  ctx.globalAlpha = 1.0;
}

function handleHover(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let hovered = null, minDist = 10;
  state.points.forEach(p => {
    const d = Math.sqrt((p.x - mx) ** 2 + (p.y - my) ** 2);
    if (d < minDist) { minDist = d; hovered = p; }
  });
  if (hovered) {
    const rect = canvas.closest('.workspace').getBoundingClientRect();
    tooltip.style.left = `${e.clientX - rect.left + 15}px`;
    tooltip.style.top = `${e.clientY - rect.top + 15}px`;
    tooltip.classList.remove('hidden');
    tooltip.innerHTML = `Income: <strong>$${hovered.valX}k</strong>, Score: <strong>${hovered.valY}</strong>`;
  } else tooltip.classList.add('hidden');
}

function saveState() {
  state.historyStack.push({ iteration: state.iteration, converged: state.converged, stepStage: state.stepStage, centroids: JSON.parse(JSON.stringify(state.centroids)), pointClusters: state.points.map(p => p.clusterIndex), text: guideText.innerHTML });
}

function undoStep() {
  if (state.historyStack.length === 0) return;
  const snap = state.historyStack.pop();
  state.iteration = snap.iteration; state.converged = snap.converged; state.stepStage = snap.stepStage;
  state.centroids = snap.centroids; guideText.innerHTML = snap.text;
  snap.pointClusters.forEach((cIdx, i) => state.points[i].clusterIndex = cIdx);
  updateSimUI(); draw();
}

function startAutoRun() { if (state.converged) return; state.isAutoRunning = true; btns.auto.innerText = "Stop"; state.animationId = setInterval(stepAlgo, 800); }
function stopAutoRun() { state.isAutoRunning = false; if (btns.auto) btns.auto.innerText = "Auto Run"; clearInterval(state.animationId); updateSimUI(); }

function updateSimUI() {
  if (!displays.iter) return;
  displays.iter.innerText = state.iteration;
  const hasCentroids = state.centroids.length > 0;
  btns.init.disabled = state.isAutoRunning && !state.converged;
  btns.prev.disabled = state.historyStack.length === 0 || state.isAutoRunning;
  btns.step.disabled = !hasCentroids || state.converged || state.isAutoRunning;
  btns.auto.disabled = !hasCentroids || state.converged;
}

function restartExperiment() {
  STATE.stepIndex = 0;
  STATE.subStepIndex = 0;
  STATE.stepsStatus = stepsData.map(() => ({ unlocked: false, completed: false, partial: false }));
  STATE.stepsStatus[0].unlocked = true;
  init();
}

function highlightCode(code) {
  return code
    .replace(/import /g, '<span class="kw">import </span>')
    .replace(/from /g, '<span class="kw">from </span>')
    .replace(/print/g, '<span class="func">print</span>')
    .replace(/def /g, '<span class="kw">def </span>')
    .replace(/return /g, '<span class="kw">return </span>');
}



function downloadPDF() {
  const link = document.createElement('a');
  link.href = './Experiment-09.pdf';
  link.download = 'Experiment-09.pdf';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}


init();
