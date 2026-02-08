export function ProgressBar({
  progress,
  width = "460px",
  height = "4px",
}: {
  progress: number;
  width?: string;
  height?: string;
}) {
  return (
    <div
      style={{
        width,
        height,
        borderRadius: "10px",
        backgroundColor: "rgba(255,255,255,0.3)",
      }}
    >
      <div
        style={{
          width: `${Math.max(0, Math.min(100, progress))}%`,
          height: "100%",
          backgroundColor: "#fff",
          borderRadius: "10px",
          transition: "width 0.35s ease",
        }}
      ></div>
    </div>
  );
}
