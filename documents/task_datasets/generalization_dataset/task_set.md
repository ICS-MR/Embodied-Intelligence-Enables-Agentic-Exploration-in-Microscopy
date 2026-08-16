# Claude Full Task Set

Source: `docs/experiment_records/01_model_comparison/claude-full/final_records`

| ID | Source | User Input |
| --- | --- | --- |
| 1 | `final_records/1.md` | Imaging target: 2D section; capture the blue fluorescent channel and process using deconvolution algorithm |
| 2 | `final_records/2.md` | Imaging target: Organoids; perform imaging of organoids in 96-well plates using a 20× objective |
| 3 | `final_records/3.md` | Imaging target: 2D section; obtain images in the current state and perform cell segmentation using Cellpose to generate segmentation masks |
| 4 | `final_records/4.md` | Imaging target: 2D section; first move to the middle of the Z-axis, capture brightfield images under a 20× objective, and perform denoising processing |
| 5 | `final_records/5.md` | Imaging target: 2D section; acquire a 3×3 mm brightfield image under a 4× objective and detect the positions of 2D cells. |
| 6 | `final_records/6.md` | Imaging target: Organoids; under blue fluorescence, capture images of organoids in 24-well plates every 1 hour using a high-magnification 20× objective for 24 hours continuously |
| 7 | `final_records/7.md` | Imaging target: 2D section; turn on the laser and simultaneously capture red fluorescent images |
| 8 | `final_records/8.md` | Imaging target: Organoids; capture images of organoids in 96-well plates every 1 hour under a 20× objective for 24 hours continuously |
| 9 | `final_records/9.md` | Imaging target: 2D section; obtain a 5×1 mm brightfield image under a 4× objective and capture the same area using blue fluorescence |
| 10 | `final_records/10.md` | Imaging target: 2D section; move the Z-axis to the middle, perform auto-focus, and return the current hardware status |
| 11 | `final_records/11.md` | [input 1] Imaging target: 2D section; switch to the 4× objective<br>[input 2] scan a 5×1 mm region to detect 2D cell areas<br>[input 3] move to the first detected region<br>[input 4] switch to the 20× objective, and acquire an image of a 3×3 mm field of view. |
| 12 | `final_records/12.md` | Imaging target: Organoids; switch to the 4脳 objective / scan a 5脳1 mm region to detect organoids / move to the first detected region / switch to the 20脳 objective, and acquire images of a 3脳3 mm field of view. |
| 13 | `final_records/13.md` | Imaging target: 2D cells; switch to the 4× objective, scan a 5×1 mm area, detect 2D cell regions, move to the first detected region, switch to the 20× objective, and acquire a 3×3 mm field of view. |
| 14 | `final_records/14.md` | Imaging target: 2D section; use the 20× objective to capture images of the 3×3 mm fluorescent slice in brightfield, blue, and green fluorescent channels, and count the size and number of cell nuclei in the blue fluorescent image. |
| 15 | `final_records/15.md` | Imaging target: Organoids; switch to low-magnification brightfield, capture the 2×2 mm array area, locate organoids based on brightfield information, switch to the 20× objective, and sequentially capture the blue, green, and red fluorescent channels of each organoid. |
| 16 | `final_records/16.md` | Imaging target: Organoids; use a low-magnification objective to locate the distribution of organoids within the field of view, then use a high-magnification objective to sequentially acquire brightfield and green fluorescence data. The operation should continue for 24 hours, with imaging performed every 30 minutes. |
| 17 | `final_records/17.md` | Imaging target: Organoids; perform brightfield scanning using the 4× objective to obtain images of the 3×3 mm area, identify and record the positions of organoids; then switch to the 20× objective, and sequentially capture images of the blue (DAPI), green (FITC), and red (TRITC) fluorescent channels of each organoid. |
| 18 | `final_records/18.md` | Imaging target: 2D section; switch to the 10× objective, perform a brightfield scan of a 4×4 mm region, detect 2D cells, move to the first detected region, switch to the 40× objective, sequentially activate the green and red fluorescence channels, acquire images of a 2×2 mm field of view, and measure the fluorescence intensity. |
| 19 | `final_records/19.md` | Imaging target: Organoids; use the low-magnification (10×) brightfield mode to scan the 4×4 mm area, detect the positions of organoids, and record their coordinates; switch to the 20× objective, and capture images of the blue and green fluorescent channels in the area of the first organoid's position. |
| 20 | `final_records/20.md` | Imaging target: 3D cells; use a 4× objective to acquire the number distribution of cells in a 2×2 mm area, which requires scanning along the Z-axis. |
| 21 | `final_records/21.md` | Imaging target: Organoids; detect organoids at the current position using the low-magnification (4×) brightfield mode and record their coordinates; switch to the 20× objective, capture images of the blue and red fluorescent channels in the area of the second organoid's position, and merge the images. |
| 22 | `final_records/22.md` | Imaging target: Organoids; first, use the 4× objective to acquire a 5×1 mm view and determine the locations of the organoids, then use the 20× objective to perform high-resolution imaging of the organoids. |
| 23 | `final_records/23.md` | Imaging target: 2D section; use a 4× objective to automatically count and locate DAPI-labeled cell nuclei on fluorescence in situ hybridization slices. |
| 24 | `final_records/24.md` | Imaging target: 2D section; simultaneously capture images of multiple fluorescent labels (stained with TMRM, Hoechst, and Calcein), and merge the different channels. |
| 25 | `final_records/25.md` | Imaging target: Organoids; continuously image the organoids in a 24-well plate for 24 hours, with imaging performed once every hour. |
| 26 | `final_records/26.md` | Imaging target: 2D section; capture images of the green fluorescent channel, apply super-resolution algorithms to resolve subcellular structures, and save the clear images. |
| 27 | `final_records/27.md` | Imaging target: 2D section; perform blue fluorescence imaging of cells in the field of view, and use algorithms to generate clear images. |
| 28 | `final_records/28.md` | [input 1] Imaging target: 2D cells; capture fluorescent channels in a 3×3 mm area to distinguish live<br>[input 2] dead cells (labeled with Calcein-AM). |
| 29 | `final_records/29.md` | Imaging target: 2D section; scan and stitch the entire image under blue fluorescence conditions, and count the number of labeled cells. |
| 30 | `final_records/30.md` | Imaging target: 2D section; detect weak blue fluorescent signals of low-expression targets. |
| 31 | `final_records/31.md` | Imaging target: Organoids; perform Z-axis imaging of organoids and synthesize clear images. |
| 32 | `final_records/32.md` | Imaging target: cell section; acquire images of the cellular state across the entire culture dish, automatically identify the cells within the dish, and perform cell counting. |
| 33 | `final_records/33.md` | Imaging target: Organoids; determine the positions of organoids using a 4× objective, then perform detailed data collection for all detected organoid regions using a 60× objective. |
| 34 | `final_records/34.md` | Imaging target: Organoids; perform continuous 24-hour imaging of the green and red fluorescent channels of organoids in a 2×2 mm gel droplet, capturing one image per hour. |
| 35 | `final_records/35.md` | Imaging target: Organoids; use a low-magnification objective to collect the positions of organoids in a 1 mm circular gel droplet, then sequentially collect the status of organoids using a 20× objective. |
| 36 | `final_records/36.md` | Imaging target: 2D section; acquire brightfield and images of different fluorescent signals, and overlay the fluorescent signals with transmitted light. |
| 37 | `final_records/37.md` | Imaging target: Organoids; perform blue and green fluorescence imaging of organoids in a 96-well plate every 1 hour. |
| 38 | `final_records/38.md` | Imaging target: 3D cells; capture images every 5 minutes for 12 hours to record the migration process of living cells. |
| 39 | `final_records/39.md` | Imaging target: Organoids; capture images every 30 minutes for 6 consecutive hours to record multiple fluorescent labels of organoids. |
| 40 | `final_records/40.md` | Imaging target: 2D section; use a low-magnification objective to complete a full scan of the entire section, automatically detect and record 2D cells, and output all detected locations. |
| 41 | `final_records/41.md` | Imaging target: Organoids; acquire brightfield and images of different fluorescent signals, and overlay the fluorescent signals with transmitted light. |
| 42 | `final_records/42.md` | Imaging target: Organoids; adjust the brightness, perform focusing, obtain the recommended Z-stack range, and move the Z-axis to the midpoint of that range. |
| 43 | `final_records/43.md` | Imaging target: 2D section; adjust the brightness to the optimal level, capture an image, then set the brightness to half of that optimal level. |
| 44 | `final_records/44.md` | Imaging target: 2D section; adjust the brightness, perform focusing, acquire a global image using a 4× objective, detect the positions of 2D cells, move to the position with the largest 2D cell area, adjust the brightness and perform focusing again, and capture an image using a 20× objective. |
| 45 | `final_records/45.md` | Imaging target: 2D section; adjust the brightness, perform focusing, capture an image, detect the positions of 2D cells, then acquire images in descending order of 2D cell area. |
| 46 | `final_records/46.md` | Imaging target: 2D section; perform focusing on the current target, then acquire images at Z-axis positions corresponding to 0.5×, 1×, and 1.5× of the current Z-axis position. |
| 47 | `final_records/47.md` | Imaging target: 2D section; adjust the brightness, perform focusing, capture the current image, automatically adjust the contrast, then display both the original and processed images simultaneously using plt for 10 seconds before closing the display. |
| 48 | `final_records/48.md` | Imaging target: 2D section; adjust the brightness, perform focusing, capture the current image, perform denoising and deconvolution, then display both the original and processed images simultaneously using plt for 10 seconds before closing the display. |
| 49 | `final_records/49.md` | Imaging target: 2D section; adjust the brightness, perform focusing, acquire an image of a 3 × 3 mm region, add a scale bar and a text label in the lower-left corner of the image, display it using plt for 10 seconds, then close the display. |
| 50 | `final_records/50.md` | Imaging target: 2D section; adjust the brightness, perform focusing, capture the current image, detect 2D cell regions, annotate the 2D cell areas with bounding boxes, and display the result using plt for 5 seconds before closing the display. |
| 51 | `final_records/51.md` | Imaging target: 3D organoids; use time to measure the procedure, adjust the brightness, perform focusing, and output the elapsed time. |
