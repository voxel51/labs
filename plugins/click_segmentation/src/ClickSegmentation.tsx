import { executeOperator } from "@fiftyone/operators";
import * as fos from "@fiftyone/state";
import _ from "lodash";
import { useRecoilValue } from "recoil";
import styled from "styled-components";
import { useState, useRef, useCallback, useEffect } from "react";

export function ClickSegmentation() {
  const modalSample = useRecoilValue(fos.modalSample);
  const [clicks, setClicks] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isNegativeClick, setIsNegativeClick] = useState(false);
  const [isSingleClick, setSingleClick] = useState(false);
  const [keypointFieldName, setKeypointFieldName] = useState("user_clicks");
  const [segFieldName, setSegFieldName] = useState("");
  const [modelName, setModelName] = useState(
    "segment-anything-2-hiera-tiny-image-torch"
  );
  const [labelName, setLabelName] = useState("label");
  const markersRef = useRef<HTMLElement[]>([]);

  useEffect(() => {
    if (!modalSample) return;
    const metadata = modalSample.sample.metadata;

    if (!metadata || !metadata.width || !metadata.height) {
      console.log("Sample metadata not available. Computing metadata");

      executeOperator("@51labs/click_segmentation/compute_metadata", {}).catch(
        (error) => {
          console.error("Failed to compute metadata:", error);
        }
      );
    }
  }, [modalSample?.sample?._id]);

  useEffect(() => {
    if (!isCapturing || !modalSample) return;

    const handleCanvasClick = async (event: MouseEvent) => {
      const canvas = event.target as HTMLCanvasElement;
      if (canvas.tagName !== "CANVAS") return;

      const rect = canvas.getBoundingClientRect();
      const metadata = modalSample.sample.metadata;

      if (!metadata || !metadata.width || !metadata.height) {
        alert(
          "Sample metadata not yet available. Reload/refresh the sample modal."
        );
        return;
      }
      // Calculate the relative position of the click
      // NOTE: Sample modal displays the image in contain mode filling one dimension while keeping the aspect
      // ratio. It is possible to zoom in and out in the model which will break the correct image coord capture.
      // TODO: Fix this for when image width and height are both smaller than that of modal canvas.
      const imageAspect = metadata.width / metadata.height;
      const canvasAspect = rect.width / rect.height;

      let imgWidth, imgHeight, imgOffsetX, imgOffsetY;

      if (imageAspect > canvasAspect) {
        // Image fills width only
        imgWidth = rect.width;
        imgHeight = rect.width / imageAspect;
        imgOffsetX = 0;
        imgOffsetY = (rect.height - imgHeight) / 2;
      } else {
        // Image fills height
        imgHeight = rect.height;
        imgWidth = rect.height * imageAspect;
        imgOffsetX = (rect.width - imgWidth) / 2;
        imgOffsetY = 0;
      }

      const canvasX = event.clientX - rect.left;
      const canvasY = event.clientY - rect.top;
      const imgX = canvasX - imgOffsetX;
      const imgY = canvasY - imgOffsetY;

      if (imgX < 0 || imgX > imgWidth || imgY < 0 || imgY > imgHeight) {
        console.log("Click outside image bounds");
        return;
      }

      const normalizedX = imgX / imgWidth;
      const normalizedY = imgY / imgHeight;
      const clickData = {
        normalizedX: parseFloat(normalizedX.toFixed(4)),
        normalizedY: parseFloat(normalizedY.toFixed(4)),
        label: isNegativeClick ? 0 : 1,
      };

      console.log("Click captured:", clickData);
      showClickMarker(event.clientX, event.clientY);

      if (isSingleClick) {
        await autoSegmentWithSingleKeypoint(clickData);
      } else {
        setClicks((prev) => [...prev, clickData]);
      }
    };

    document.addEventListener("click", handleCanvasClick, true);
    return () => document.removeEventListener("click", handleCanvasClick, true);
  }, [
    isCapturing,
    modalSample,
    isSingleClick,
    isNegativeClick,
    modelName,
    keypointFieldName,
  ]);

  const saveAsKeypoints = async () => {
    if (!keypointFieldName.trim()) {
      alert("Please enter a field name");
      return;
    }

    const keypointCoords = clicks.map((click) => [
      click.normalizedX,
      click.normalizedY,
    ]);

    const keypointLabels = clicks.map((click) => [click.label]);

    try {
      await executeOperator("@51labs/click_segmentation/save_keypoints", {
        keypoints: keypointCoords,
        keypoint_labels: keypointLabels,
        kpts_field_name: keypointFieldName.trim(),
        label_name: labelName.trim(),
      });
      await executeOperator("reload_dataset");
    } catch (error) {
      console.error("Error saving keypoints:", error);
      alert(`Failed: ${error.message}`);
    } finally {
      clearClicks();
    }
  };

  const autoSegmentWithSingleKeypoint = async (clickData) => {
    const keypointCoords = [[clickData.normalizedX, clickData.normalizedY]];
    try {
      alert("Segmentation in progress");
      await executeOperator("@51labs/click_segmentation/save_keypoints", {
        keypoints: keypointCoords,
        kpts_field_name: keypointFieldName.trim(),
        label_name: labelName.trim(),
      });
      await executeOperator("@51labs/click_segmentation/segment_with_prompts", {
        prompt_field: keypointFieldName.trim(),
        model_name: modelName.trim(),
        label_field: segFieldName.trim(),
      });
      setClicks([]);
    } catch (error) {
      console.error("Error auto segmenting:", error);
      alert(`Failed: ${error.message}`);
    } finally {
      clearClicks();
    }
  };

  const segmentWithKeypoints = async () => {
    try {
      if (!isSingleClick) {
        alert("Segmentation in progress");
      }
      await executeOperator("@51labs/click_segmentation/segment_with_prompts", {
        prompt_field: keypointFieldName.trim(),
        model_name: modelName.trim(),
        label_field: segFieldName.trim(),
      });
    } catch (error) {
      console.error("Error saving keypoints:", error);
      alert(`Failed: ${error.message}`);
    }
  };

  const showClickMarker = (x, y) => {
    const marker = document.createElement("div");
    marker.className = "click-marker";
    marker.style.position = "fixed";
    marker.style.left = `${x}px`;
    marker.style.top = `${y}px`;
    marker.style.width = "12px";
    marker.style.height = "12px";
    marker.style.borderRadius = "50%";
    marker.style.backgroundColor = isNegativeClick ? "#1da10e" : "#ff4444";
    marker.style.border = "3px solid white";
    marker.style.transform = "translate(-50%, -50%)";
    marker.style.pointerEvents = "none";
    marker.style.zIndex = "99999";
    marker.style.boxShadow = "0 0 10px rgba(255,0,0,0.8)";

    document.body.appendChild(marker);
    markersRef.current.push(marker);
  };

  const clearAllMarkers = () => {
    markersRef.current.forEach((marker) => {
      marker.style.transition =
        "opacity 0.3s ease-out, transform 0.3s ease-out";
      marker.style.opacity = "0";
      marker.style.transform = "translate(-50%, -50%) scale(0)";
      setTimeout(() => {
        marker.remove();
      }, 300);
    });

    markersRef.current = [];
  };

  const clearClicks = () => {
    console.log("Clearing clicks and markers");
    setClicks([]);
    clearAllMarkers();
  };

  return (
    <div
      style={{
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        height: "100%",
      }}
    >
      <div style={{ marginBottom: "15px" }}>
        <h3 style={{ margin: 0 }}>Image segmentation via point prompts</h3>
      </div>

      {/* Capture toggle on/off*/}
      <div
        style={{
          marginBottom: "20px",
          display: "flex",
          flexDirection: "column",
          padding: "15px",
          backgroundColor: isCapturing ? "#e3f2fd" : "#f5f5f5",
          borderRadius: "8px",
          border: `2px solid ${isCapturing ? "#2196F3" : "#ddd"}`,
          transition: "all 0.3s ease",
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={isCapturing}
            onChange={(e) => setIsCapturing(e.target.checked)}
            style={{
              width: "20px",
              height: "20px",
              cursor: "pointer",
            }}
          />
          <span
            style={{
              fontWeight: "bold",
              fontSize: "15px",
              color: isCapturing ? "#1976d2" : "#666",
            }}
          >
            {isCapturing ? "Capturing Clicks" : "Activate Click Capture"}
          </span>
        </label>
        {isCapturing && (
          <p
            style={{
              fontSize: "12px",
              color: "#1976d2",
              margin: "8px 0 0 30px",
              fontWeight: "500",
            }}
          >
            Click anywhere on the modal image to capture keypoints.
          </p>
        )}
      </div>

      {/* Keypoint field name input */}
      <div
        style={{
          marginBottom: "8px",
          padding: "4px",
        }}
      >
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            fontSize: "14px",
          }}
        >
          <span style={{ fontSize: "14px", color: "#666" }}>
            Sample field name for saving clicks (as keypoints)
          </span>
          <input
            type="text"
            value={keypointFieldName}
            onChange={(e) => setKeypointFieldName(e.target.value)}
            placeholder="e.g., user_clicks, keypoints"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace",
            }}
          />
        </label>
      </div>

      {/* Keypoint label name input */}
      <div
        style={{
          marginBottom: "8px",
          padding: "4px",
        }}
      >
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            fontSize: "14px",
          }}
        >
          <span style={{ fontSize: "14px", color: "#666" }}>
            Label name for the current set of clicks
          </span>
          <input
            type="text"
            value={labelName}
            onChange={(e) => setLabelName(e.target.value)}
            placeholder="e.g., animal, person"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace",
            }}
          />
        </label>
      </div>

      {/* Negative keypoint toggle on/off */}
      <div
        style={{
          marginBottom: "20px",
          display: "flex",
          flexDirection: "column",
          padding: "15px",
          backgroundColor: isNegativeClick ? "#fdf0e3" : "#f5f5f5",
          borderRadius: "8px",
          border: `2px solid ${isNegativeClick ? "#f36b21" : "#ddd"}`,
          transition: "all 0.3s ease",
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={isNegativeClick}
            onChange={(e) => {
              setSingleClick(false);
              setIsNegativeClick(e.target.checked);
            }}
            disabled={!isCapturing}
            style={{
              width: "20px",
              height: "20px",
              cursor: isCapturing ? "pointer" : "not-allowed",
            }}
          />
          <span
            style={{
              fontWeight: "bold",
              fontSize: "15px",
              color: isNegativeClick ? "#d26319" : "#666",
            }}
          >
            {isNegativeClick
              ? "Negative Clicks Enabled"
              : "Enable Negative Clicks"}
          </span>
        </label>
        {isCapturing && isNegativeClick && (
          <p
            style={{
              fontSize: "12px",
              color: "#d26319",
              margin: "8px 0 0 30px",
              fontWeight: "500",
            }}
          >
            Capturing negative keypoint prompts via clicks. Not supported for
            single click execution.
          </p>
        )}
      </div>

      {/* Keypoint buttons */}
      <div style={{ marginBottom: "10px", display: "flex", gap: "10px" }}>
        <button
          onClick={saveAsKeypoints}
          disabled={isSingleClick || clicks.length === 0}
          style={{
            padding: "8px 16px",
            backgroundColor:
              !isSingleClick && clicks.length > 0 ? "#2196F3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor:
              !isSingleClick && clicks.length > 0 ? "pointer" : "not-allowed",
            fontWeight: "bold",
          }}
        >
          Save as Keypoints ({clicks.length})
        </button>
        <button
          onClick={clearClicks}
          disabled={isSingleClick || clicks.length === 0}
          style={{
            padding: "8px 16px",
            backgroundColor:
              !isSingleClick || clicks.length > 0 ? "#2196F3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor:
              !isSingleClick && clicks.length > 0 ? "pointer" : "not-allowed",
          }}
        >
          Clear Clicks ({clicks.length})
        </button>
      </div>

      <hr
        style={{
          border: "none",
          borderTop: "1px solid #ddd",
          margin: "20px 0",
        }}
      />

      {/* Single click execution */}
      <div
        style={{
          marginBottom: "20px",
          display: "flex",
          flexDirection: "column",
          padding: "15px",
          backgroundColor: isSingleClick ? "#e3f2fd" : "#f5f5f5",
          borderRadius: "8px",
          border: `2px solid ${isSingleClick ? "#2196F3" : "#ddd"}`,
          transition: "all 0.3s ease",
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={isSingleClick}
            onChange={(e) => {
              setSingleClick(e.target.checked);
              setIsNegativeClick(false);
            }}
            disabled={!isCapturing}
            style={{
              width: "20px",
              height: "20px",
              cursor: isCapturing ? "pointer" : "not-allowed",
            }}
          />
          <span
            style={{
              fontWeight: "bold",
              fontSize: "15px",
              color: isSingleClick ? "#1976d2" : "#666",
            }}
          >
            {isSingleClick
              ? "Single Click Segmentation"
              : "Enable Single Click Segmentation"}
          </span>
        </label>
        {isSingleClick && (
          <p
            style={{
              fontSize: "12px",
              color: "#1976d2",
              margin: "8px 0 0 30px",
              fontWeight: "500",
            }}
          >
            Single click segmentation is enabled. Clicking on the modal will
            auto-generate segmentation.
          </p>
        )}
      </div>

      {/* Model name input */}
      <div
        style={{
          marginBottom: "8px",
          padding: "4px",
        }}
      >
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            fontSize: "14px",
          }}
        >
          <span style={{ fontSize: "14px", color: "#666" }}>
            Promptable segmentation model from FiftyOne model zoo
          </span>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., segment-anything-2-hiera-small-image-torch"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace",
            }}
          />
        </label>
      </div>

      {/* Segmentation field name input */}
      <div
        style={{
          marginBottom: "8px",
          padding: "4px",
        }}
      >
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            fontSize: "14px",
          }}
        >
          <span style={{ fontSize: "14px", color: "#666" }}>
            Sample field name for saving segmentation masks (optional)
          </span>
          <input
            type="text"
            value={segFieldName}
            onChange={(e) => setSegFieldName(e.target.value)}
            placeholder="Defaults to keypoint_field_seg, e.g., user_clicks_seg"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace",
            }}
          />
        </label>
      </div>

      {/* Segment with keypoints button */}
      <div style={{ marginBottom: "10px", display: "flex", gap: "10px" }}>
        <button
          onClick={segmentWithKeypoints}
          disabled={isSingleClick}
          style={{
            padding: "8px 16px",
            backgroundColor: !isSingleClick ? "#2196F3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: !isSingleClick ? "pointer" : "not-allowed",
            fontWeight: "bold",
          }}
        >
          Segment with keypoints
        </button>
      </div>
    </div>
  );
}
