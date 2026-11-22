/* ContextDepth project page: numbers from the paper (delta_1 accuracy, %).
 * Table 1: 3,406 images, six models, cropped and full-image queries, with and without GT camera.
 * Table 2: 2,635 non-DIODE queries, three reference banks.
 * Fig. 3 (top): marginal gain from restoring real context, Q0 -> Q4.
 */
window.CD_RESULTS = {
  models: [
    { key: 'moge3', name: 'MoGe-3' },
    { key: 'da3', name: 'Depth Anything 3', short: 'DA3' },
    { key: 'moge2', name: 'MoGe-2' },
    { key: 'unidepth', name: 'UniDepth V2' },
    { key: 'unik3d', name: 'UniK3D' },
    { key: 'depth_pro', name: 'Depth Pro', note: true }
  ],
  // [Query-only, Outpaint (seed 0), Outpaint + Select (mixed bank, N = 3)]
  main: {
    unknown: {
      crop: { moge3: [62.67, 76.73, 78.99], da3: [51.17, 68.47, 73.82], moge2: [60.32, 77.06, 78.28], unidepth: [62.06, 69.91, 72.04], unik3d: [62.80, 74.68, 76.37], depth_pro: [55.62, 57.40, 60.00] },
      full: { moge3: [79.63, 84.77, 85.40], da3: [73.06, 75.69, 78.67], moge2: [78.35, 84.07, 84.64], unidepth: [69.49, 71.23, 72.24], unik3d: [80.85, 81.03, 81.58], depth_pro: [62.09, 48.48, 49.85] }
    },
    known: {
      crop: { moge3: [78.36, 78.84, 81.21], da3: [73.99, 79.64, 81.05], moge2: [80.39, 82.02, 82.83], unidepth: [73.01, 72.85, 75.56], unik3d: [76.26, 76.74, 78.62], depth_pro: [68.12, 67.75, 70.56] },
      full: { moge3: [84.83, 85.81, 86.53], da3: [84.17, 85.21, 86.65], moge2: [86.77, 87.36, 87.57], unidepth: [77.37, 77.28, 79.92], unik3d: [83.94, 82.73, 84.87], depth_pro: [67.69, 66.59, 68.97] }
    }
  },
  // [Query-only, Outpaint, Select: Mixed bank, Cross-dataset bank, Fixed-DIODE bank]
  banks: {
    unknown: {
      crop: { moge3: [63.21, 79.73, 82.23, 80.27, 80.59], da3: [54.59, 70.53, 76.70, 72.38, 71.72], moge2: [62.63, 80.77, 82.04, 80.38, 80.81], unidepth: [69.04, 76.24, 78.69, 75.95, 75.92], unik3d: [66.56, 78.34, 80.08, 77.44, 77.77], depth_pro: [63.20, 62.80, 65.50, 61.72, 61.96] },
      full: { moge3: [81.71, 86.84, 87.74, 87.05, 87.16], da3: [75.72, 78.59, 81.86, 77.92, 78.57], moge2: [81.24, 86.16, 86.87, 86.17, 86.44], unidepth: [74.60, 74.27, 75.55, 74.22, 73.90], unik3d: [83.18, 82.43, 83.00, 82.84, 82.63], depth_pro: [69.08, 50.92, 52.04, 50.88, 50.90] }
    },
    known: {
      crop: { moge3: [80.84, 81.46, 84.16, 83.10, 83.80], da3: [76.40, 82.16, 84.47, 81.61, 81.91], moge2: [83.56, 85.51, 86.31, 85.06, 85.56], unidepth: [78.55, 77.39, 81.13, 75.89, 76.15], unik3d: [80.16, 79.64, 82.01, 77.57, 78.06], depth_pro: [76.38, 74.69, 77.46, 78.26, 78.27] },
      full: { moge3: [86.78, 87.36, 88.42, 87.87, 88.23], da3: [86.39, 87.94, 89.75, 87.88, 87.86], moge2: [88.80, 89.28, 89.67, 89.06, 89.57], unidepth: [80.32, 80.23, 83.47, 80.24, 79.66], unik3d: [84.75, 83.96, 86.32, 85.18, 85.11], depth_pro: [74.79, 73.28, 75.62, 76.23, 76.41] }
    }
  },
  // Marginal delta_1 gain (points) when the visible region grows: Q0->Q1, Q1->Q2, Q2->Q3, Q3->Q4.
  // Q0..Q4 span 66.7%, 75%, 83.3%, 91.7% and 100% of each image dimension (real context, no generation).
  probe: {
    steps: ['66.7 → 75', '75 → 83.3', '83.3 → 91.7', '91.7 → 100'],
    known: { moge3: [2.01, 1.75, 1.83, 1.48], da3: [3.62, 3.60, 2.48, 1.66], moge2: [1.95, 2.42, 1.41, 1.46], unidepth: [2.29, 2.25, 1.02, 0.68], unik3d: [3.72, 2.21, 0.66, 1.77], depth_pro: [0.58, 0.26, 0.58, -0.35] },
    unknown: { moge3: [6.56, 6.50, 3.73, 3.47], da3: [6.98, 4.89, 5.69, 3.25], moge2: [5.91, 6.42, 4.81, 5.31], unidepth: [4.58, 2.55, 1.21, -0.01], unik3d: [6.69, 3.55, 1.34, 0.31], depth_pro: [8.19, 1.75, -0.74, -1.91] }
  }
};
