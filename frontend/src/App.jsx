import { useState, useEffect, useRef, useCallback } from "react";

// ── Responsive helper ──────────────────────────────────────────────────────
function useIsMobile(bp = 768) {
  const [mobile, setMobile] = useState(() => typeof window !== "undefined" && window.innerWidth <= bp);
  useEffect(() => {
    const fn = () => setMobile(window.innerWidth <= bp);
    window.addEventListener("resize", fn);
    return () => window.removeEventListener("resize", fn);
  }, [bp]);
  return mobile;
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const API = import.meta.env.VITE_API_URL;

const TASKS = [
  { id:"detect",   image:"/images/task_detect.png", label:"Object Detection",        short:"Detect",
    desc:"Bounding boxes around objects in real time.",
    usecase:"Autonomous driving · Surveillance · Retail",
    label_fmt:"class_id  cx  cy  width  height  (normalized 0–1)",
    folder_hint:"images/ + labels/ folders, matching .txt files" },
  { id:"segment",  image:"/images/task_segment.png", label:"Instance Segmentation",   short:"Segment",
    desc:"Pixel-precise masks per object instance.",
    usecase:"Medical imaging · AR/VR · Satellites",
    label_fmt:"class_id  cx  cy  w  h  px1 py1 px2 py2 …  (polygon)",
    folder_hint:"YOLO structure + polygon keypoints in labels" },
  { id:"classify", image:"/images/task_classify.png", label:"Image Classification",    short:"Classify",
    desc:"Categorize the whole image into a class.",
    usecase:"Quality inspection · Diagnosis · Moderation",
    label_fmt:"No label files — class-named sub-folders",
    folder_hint:"images/train/class_name/image.jpg" },
  { id:"pose",     image:"/images/task_pose.png", label:"Pose Estimation",         short:"Pose",
    desc:"Keypoints on objects (joints, landmarks).",
    usecase:"Sports analytics · Rehabilitation · Gestures",
    label_fmt:"class_id  cx  cy  w  h  kpx kpy kpv … per keypoint",
    folder_hint:"YOLO structure; kp visibility 0=hidden 1=labeled 2=visible" },
  { id:"obb",      image:"/images/task_obb.png", label:"Oriented Bounding Boxes", short:"OBB",
    desc:"Rotated boxes for angled objects.",
    usecase:"Aerial imagery · Document scanning · PCBs",
    label_fmt:"class_id  cx  cy  width  height  angle (°, –180 to 180)",
    folder_hint:"YOLO detect structure + angle column" },
];

const FORMAT_GUIDE = {
  yolo:{ label:"YOLO Format", color:"#3b82f6",
    structure:`dataset/\n├── images/\n│   ├── train/    ← .jpg/.png images\n│   ├── val/\n│   └── test/\n├── labels/\n│   ├── train/    ← .txt (same name)\n│   ├── val/\n│   └── test/\n└── data.yaml`,
    labels:`# one line per object\nclass_id  cx  cy  width  height\n\n# Example — person at center\n0  0.50  0.50  0.48  0.65\n1  0.23  0.41  0.18  0.32`,
    yaml:`path:  ./dataset\ntrain: images/train\nval:   images/val\ntest:  images/test\nnc:    3\nnames: ['cat','dog','bird']`,
    notes:`• All coords normalized [0,1]\n• cx,cy = box centre (not top-left)\n• Empty .txt = no objects\n• class_id is 0-indexed\n• filenames must match exactly` },
  coco:{ label:"COCO JSON", color:"#06b6d4",
    structure:`dataset/\n├── images/\n│   ├── train2024/\n│   └── val2024/\n└── annotations/\n    ├── instances_train2024.json\n    └── instances_val2024.json`,
    labels:`{\n  "images":[{"id":1,"file_name":"img.jpg"}],\n  "annotations":[{\n    "id":1,"image_id":1,"category_id":1,\n    "bbox":[x_tl,y_tl,w,h],"area":1234\n  }],\n  "categories":[{"id":1,"name":"cat"}]\n}`,
    yaml:`# pip install pycocotools\npath:  ./dataset\ntrain: images/train2024\nval:   images/val2024\nnc:    80\nnames: ['person','car',...]`,
    notes:`• COCO bbox = pixel coords [x_tl,y_tl,w,h]\n• Needs: pip install pycocotools\n• Ultralytics auto-converts internally\n• category_id must be consistent` },
  voc:{ label:"Pascal VOC XML", color:"#f59e0b",
    structure:`dataset/\n├── JPEGImages/      ← .jpg images\n├── Annotations/     ← .xml files\n└── ImageSets/Main/\n    ├── train.txt\n    └── val.txt`,
    labels:`<annotation>\n  <filename>img.jpg</filename>\n  <size><width>640</width><height>480</height></size>\n  <object>\n    <name>cat</name>\n    <bndbox>\n      <xmin>120</xmin><ymin>80</ymin>\n      <xmax>320</xmax><ymax>280</ymax>\n    </bndbox>\n  </object>\n</annotation>`,
    yaml:`# Convert to YOLO first:\n# python ultralytics/data/converter.py\n# Or export as YOLO from LabelImg`,
    notes:`• Absolute pixel coords\n• One .xml per image\n• <name> must be consistent\n• Best: convert to YOLO before training` },
  roboflow:{ label:"Roboflow Export", color:"#8b5cf6",
    structure:`export/\n├── train/\n│   ├── images/\n│   └── labels/\n├── valid/\n│   ├── images/\n│   └── labels/\n├── test/\n│   ├── images/\n│   └── labels/\n└── data.yaml  ← use directly`,
    labels:`# Same as YOLO format\nclass_id  cx  cy  width  height\n\n# data.yaml pre-filled by Roboflow\n# Point --data directly to it`,
    yaml:`path:  /path/to/export\ntrain: train/images\nval:   valid/images\ntest:  test/images\nnc:    5\nnames: [...]`,
    notes:`• Export as "YOLOv8" format\n• data.yaml is pre-filled — use directly\n• Split already done for you\n• Augmentations stack with Roboflow's` },
};

const ALL_MODELS = [
  {id:"yolov5nu",  n:"YOLOv5n",       p:"1.9M",  s:"3.8MB",  m:"28.0",fps:45,  f:"v5",     t:["detect"],                              src:"ultralytics",pt:"yolov5nu.pt",    y:"yolov5n.yaml"},
  {id:"yolov5su",  n:"YOLOv5s",       p:"7.2M",  s:"14MB",   m:"37.4",fps:38,  f:"v5",     t:["detect"],                              src:"ultralytics",pt:"yolov5su.pt",    y:"yolov5s.yaml"},
  {id:"yolov5mu",  n:"YOLOv5m",       p:"21.2M", s:"41MB",   m:"45.4",fps:28,  f:"v5",     t:["detect"],                              src:"ultralytics",pt:"yolov5mu.pt",    y:"yolov5m.yaml"},
  {id:"yolov5lu",  n:"YOLOv5l",       p:"46.5M", s:"89MB",   m:"49.0",fps:20,  f:"v5",     t:["detect"],                              src:"ultralytics",pt:"yolov5lu.pt",    y:"yolov5l.yaml"},
  {id:"yolov5xu",  n:"YOLOv5x",       p:"86.7M", s:"166MB",  m:"50.7",fps:15,  f:"v5",     t:["detect"],                              src:"ultralytics",pt:"yolov5xu.pt",    y:"yolov5x.yaml"},
  {id:"yolov8n",   n:"YOLOv8n",       p:"3.2M",  s:"6MB",    m:"37.3",fps:80,  f:"v8",     t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolov8n.pt",  y:"yolov8n.yaml"},
  {id:"yolov8s",   n:"YOLOv8s",       p:"11.2M", s:"22MB",   m:"44.9",fps:65,  f:"v8",     t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolov8s.pt",  y:"yolov8s.yaml"},
  {id:"yolov8m",   n:"YOLOv8m",       p:"25.9M", s:"50MB",   m:"50.2",fps:47,  f:"v8",     t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolov8m.pt",  y:"yolov8m.yaml"},
  {id:"yolov8l",   n:"YOLOv8l",       p:"43.7M", s:"84MB",   m:"52.9",fps:33,  f:"v8",     t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolov8l.pt",  y:"yolov8l.yaml"},
  {id:"yolov8x",   n:"YOLOv8x",       p:"68.2M", s:"130MB",  m:"53.9",fps:25,  f:"v8",     t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolov8x.pt",  y:"yolov8x.yaml"},
  {id:"yolov8n-seg",n:"YOLOv8n-seg",  p:"3.4M",  s:"7MB",    m:"36.7",fps:75,  f:"v8",     t:["segment"],                             src:"ultralytics",pt:"yolov8n-seg.pt",y:"yolov8n-seg.yaml"},
  {id:"yolov8s-seg",n:"YOLOv8s-seg",  p:"11.8M", s:"23MB",   m:"44.6",fps:60,  f:"v8",     t:["segment"],                             src:"ultralytics",pt:"yolov8s-seg.pt",y:"yolov8s-seg.yaml"},
  {id:"yolov8m-seg",n:"YOLOv8m-seg",  p:"27.3M", s:"52MB",   m:"49.9",fps:43,  f:"v8",     t:["segment"],                             src:"ultralytics",pt:"yolov8m-seg.pt",y:"yolov8m-seg.yaml"},
  {id:"yolov8n-cls",n:"YOLOv8n-cls",  p:"2.7M",  s:"5MB",    m:"69.0",fps:120, f:"v8",     t:["classify"],                            src:"ultralytics",pt:"yolov8n-cls.pt",y:"yolov8n-cls.yaml"},
  {id:"yolov8s-cls",n:"YOLOv8s-cls",  p:"6.4M",  s:"13MB",   m:"73.8",fps:95,  f:"v8",     t:["classify"],                            src:"ultralytics",pt:"yolov8s-cls.pt",y:"yolov8s-cls.yaml"},
  {id:"yolov8m-cls",n:"YOLOv8m-cls",  p:"17.0M", s:"33MB",   m:"76.8",fps:65,  f:"v8",     t:["classify"],                            src:"ultralytics",pt:"yolov8m-cls.pt",y:"yolov8m-cls.yaml"},
  {id:"yolov8n-pose",n:"YOLOv8n-pose",p:"3.3M",  s:"6MB",    m:"50.4",fps:80,  f:"v8",     t:["pose"],                                src:"ultralytics",pt:"yolov8n-pose.pt",y:"yolov8n-pose.yaml"},
  {id:"yolov8s-pose",n:"YOLOv8s-pose",p:"11.6M", s:"23MB",   m:"60.0",fps:62,  f:"v8",     t:["pose"],                                src:"ultralytics",pt:"yolov8s-pose.pt",y:"yolov8s-pose.yaml"},
  {id:"yolov8m-pose",n:"YOLOv8m-pose",p:"26.4M", s:"51MB",   m:"65.0",fps:44,  f:"v8",     t:["pose"],                                src:"ultralytics",pt:"yolov8m-pose.pt",y:"yolov8m-pose.yaml"},
  {id:"yolov8n-obb",n:"YOLOv8n-obb",  p:"3.1M",  s:"6MB",    m:"78.0",fps:80,  f:"v8",     t:["obb"],                                 src:"ultralytics",pt:"yolov8n-obb.pt", y:"yolov8n-obb.yaml"},
  {id:"yolov8s-obb",n:"YOLOv8s-obb",  p:"11.4M", s:"22MB",   m:"79.5",fps:62,  f:"v8",     t:["obb"],                                 src:"ultralytics",pt:"yolov8s-obb.pt", y:"yolov8s-obb.yaml"},
  {id:"yolov9t",   n:"YOLOv9t",       p:"2.0M",  s:"3.9MB",  m:"38.3",fps:78,  f:"v9",     t:["detect","segment"],                    src:"ultralytics",pt:"yolov9t.pt",     y:"yolov9t.yaml"},
  {id:"yolov9s",   n:"YOLOv9s",       p:"7.2M",  s:"14MB",   m:"46.8",fps:60,  f:"v9",     t:["detect","segment"],                    src:"ultralytics",pt:"yolov9s.pt",     y:"yolov9s.yaml"},
  {id:"yolov9c",   n:"YOLOv9c",       p:"25.3M", s:"49MB",   m:"53.0",fps:38,  f:"v9",     t:["detect","segment"],                    src:"ultralytics",pt:"yolov9c.pt",     y:"yolov9c.yaml"},
  {id:"yolov9e",   n:"YOLOv9e",       p:"57.3M", s:"113MB",  m:"55.6",fps:22,  f:"v9",     t:["detect","segment"],                    src:"ultralytics",pt:"yolov9e.pt",     y:"yolov9e.yaml"},
  {id:"yolov10n",  n:"YOLOv10n",      p:"2.7M",  s:"5MB",    m:"39.5",fps:90,  f:"v10",    t:["detect","segment"],                    src:"ultralytics",pt:"yolov10n.pt",    y:"yolov10n.yaml"},
  {id:"yolov10s",  n:"YOLOv10s",      p:"8.0M",  s:"16MB",   m:"46.7",fps:70,  f:"v10",    t:["detect","segment"],                    src:"ultralytics",pt:"yolov10s.pt",    y:"yolov10s.yaml"},
  {id:"yolov10m",  n:"YOLOv10m",      p:"16.5M", s:"32MB",   m:"51.3",fps:51,  f:"v10",    t:["detect","segment"],                    src:"ultralytics",pt:"yolov10m.pt",    y:"yolov10m.yaml"},
  {id:"yolov10l",  n:"YOLOv10l",      p:"25.8M", s:"50MB",   m:"53.4",fps:38,  f:"v10",    t:["detect","segment"],                    src:"ultralytics",pt:"yolov10l.pt",    y:"yolov10l.yaml"},
  {id:"yolov10x",  n:"YOLOv10x",      p:"31.6M", s:"61MB",   m:"54.4",fps:30,  f:"v10",    t:["detect","segment"],                    src:"ultralytics",pt:"yolov10x.pt",    y:"yolov10x.yaml"},
  {id:"yolo11n",   n:"YOLO11n",       p:"2.6M",  s:"5MB",    m:"39.5",fps:90,  f:"v11",    t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolo11n.pt",  y:"yolo11n.yaml"},
  {id:"yolo11s",   n:"YOLO11s",       p:"9.4M",  s:"18MB",   m:"47.0",fps:68,  f:"v11",    t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolo11s.pt",  y:"yolo11s.yaml"},
  {id:"yolo11m",   n:"YOLO11m",       p:"20.1M", s:"39MB",   m:"51.5",fps:46,  f:"v11",    t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolo11m.pt",  y:"yolo11m.yaml"},
  {id:"yolo11l",   n:"YOLO11l",       p:"25.3M", s:"49MB",   m:"53.4",fps:36,  f:"v11",    t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolo11l.pt",  y:"yolo11l.yaml"},
  {id:"yolo11x",   n:"YOLO11x",       p:"56.9M", s:"109MB",  m:"54.7",fps:24,  f:"v11",    t:["detect","segment","classify","pose","obb"],src:"ultralytics",pt:"yolo11x.pt",  y:"yolo11x.yaml"},
  {id:"yolo11n-seg",n:"YOLO11n-seg",  p:"2.9M",  s:"6MB",    m:"38.9",fps:80,  f:"v11",    t:["segment"],                             src:"ultralytics",pt:"yolo11n-seg.pt",y:"yolo11n-seg.yaml"},
  {id:"yolo11n-cls",n:"YOLO11n-cls",  p:"1.6M",  s:"3MB",    m:"70.0",fps:130, f:"v11",    t:["classify"],                            src:"ultralytics",pt:"yolo11n-cls.pt",y:"yolo11n-cls.yaml"},
  {id:"yolo11n-pose",n:"YOLO11n-pose",p:"2.9M",  s:"6MB",    m:"51.1",fps:88,  f:"v11",    t:["pose"],                                src:"ultralytics",pt:"yolo11n-pose.pt",y:"yolo11n-pose.yaml"},
  {id:"yolo11n-obb",n:"YOLO11n-obb",  p:"2.7M",  s:"5MB",    m:"78.4",fps:88,  f:"v11",    t:["obb"],                                 src:"ultralytics",pt:"yolo11n-obb.pt", y:"yolo11n-obb.yaml"},
  {id:"yolo26n",   n:"YOLO26n",       p:"1.3M",  s:"2.8MB",  m:"36.5",fps:110, f:"v26",    t:["detect"],                              src:"ultralytics",pt:"yolo26n.pt",     y:"yolo26n.yaml"},
  {id:"yolo26s",   n:"YOLO26s",       p:"4.6M",  s:"9.5MB",  m:"44.2",fps:85,  f:"v26",    t:["detect"],                              src:"ultralytics",pt:"yolo26s.pt",     y:"yolo26s.yaml"},
  {id:"yolo26m",   n:"YOLO26m",       p:"9.6M",  s:"19MB",   m:"49.0",fps:62,  f:"v26",    t:["detect"],                              src:"ultralytics",pt:"yolo26m.pt",     y:"yolo26m.yaml"},
  {id:"rtdetr-l",  n:"RT-DETR L",     p:"32M",   s:"62MB",   m:"53.0",fps:35,  f:"rtdetr", t:["detect"],                              src:"ultralytics",pt:"rtdetr-l.pt",    y:null},
  {id:"rtdetr-x",  n:"RT-DETR X",     p:"67M",   s:"128MB",  m:"54.8",fps:22,  f:"rtdetr", t:["detect"],                              src:"ultralytics",pt:"rtdetr-x.pt",    y:null},
  {id:"yolo-world-s",n:"YOLO-World S",p:"13M",   s:"26MB",   m:"37.4",fps:55,  f:"world",  t:["detect"],                              src:"tencent",    pt:"yolov8s-worldv2.pt",y:null},
  {id:"yolo-world-m",n:"YOLO-World M",p:"29M",   s:"56MB",   m:"43.0",fps:38,  f:"world",  t:["detect"],                              src:"tencent",    pt:"yolov8m-worldv2.pt",y:null},
  {id:"yolo-world-l",n:"YOLO-World L",p:"49M",   s:"95MB",   m:"45.7",fps:28,  f:"world",  t:["detect"],                              src:"tencent",    pt:"yolov8l-worldv2.pt",y:null},
  {id:"yolo-nas-s",n:"YOLO-NAS S",    p:"12.9M", s:"25MB",   m:"47.5",fps:58,  f:"nas",    t:["detect"],                              src:"deci",       pt:"yolo_nas_s.pt",    y:null},
  {id:"yolo-nas-m",n:"YOLO-NAS M",    p:"31.9M", s:"62MB",   m:"51.5",fps:40,  f:"nas",    t:["detect"],                              src:"deci",       pt:"yolo_nas_m.pt",    y:null},
  {id:"yolo-nas-l",n:"YOLO-NAS L",    p:"42.2M", s:"81MB",   m:"52.2",fps:30,  f:"nas",    t:["detect"],                              src:"deci",       pt:"yolo_nas_l.pt",    y:null},
];

const AUG_GROUPS = [
  { title:"🎨 Color",    gc:"#f59e0b", items:[
    {k:"hsv_h",l:"Hue Shift",    tip:"Shifts hue channel",           min:0,max:1,    step:0.005,def:0.015,defOn:true},
    {k:"hsv_s",l:"Saturation",   tip:"Random saturation",            min:0,max:1,    step:0.05, def:0.7,  defOn:true},
    {k:"hsv_v",l:"Brightness",   tip:"Random brightness",            min:0,max:1,    step:0.05, def:0.4,  defOn:true},
  ]},
  { title:"📐 Geometry", gc:"#3b82f6", items:[
    {k:"degrees",  l:"Rotation",    tip:"Random rotation ±°",         min:0,max:45,   step:1,    def:0,    defOn:false},
    {k:"translate",l:"Translation", tip:"Random XY shift",            min:0,max:0.5,  step:0.01, def:0.1,  defOn:true},
    {k:"scale",    l:"Scale",       tip:"Random zoom",                min:0,max:0.9,  step:0.05, def:0.5,  defOn:true},
    {k:"shear",    l:"Shear",       tip:"Horizontal shear",           min:0,max:20,   step:0.5,  def:0,    defOn:false},
    {k:"perspective",l:"Perspective",tip:"Perspective warp",         min:0,max:0.001,step:0.0001,def:0,   defOn:false},
  ]},
  { title:"↔️ Flip",     gc:"#10b981", items:[
    {k:"fliplr",l:"Horiz. Flip", tip:"Left-right flip prob",          min:0,max:1,    step:0.05, def:0.5,  defOn:true},
    {k:"flipud",l:"Vert. Flip",  tip:"Up-down flip prob",             min:0,max:1,    step:0.05, def:0,    defOn:false},
  ]},
  { title:"🎭 Mix",      gc:"#8b5cf6", items:[
    {k:"mosaic",      l:"Mosaic",       tip:"4-image composite",      min:0,max:1,    step:0.05, def:1.0,  defOn:true},
    {k:"mixup",       l:"MixUp",        tip:"Alpha-blend 2 images",   min:0,max:1,    step:0.05, def:0,    defOn:false},
    {k:"copy_paste",  l:"Copy-Paste",   tip:"Paste objects across",   min:0,max:1,    step:0.05, def:0,    defOn:false},
    {k:"erasing",     l:"Rand. Erase",  tip:"Erase random rects",    min:0,max:0.9,  step:0.05, def:0,    defOn:false},
    {k:"crop_fraction",l:"Crop Frac",   tip:"Crop image fraction",   min:0.1,max:1,  step:0.05, def:1.0,  defOn:false},
  ]},
];

const SCENES = [
  {id:"street",  label:"Street",   emoji:"🏙️"},
  {id:"aerial",  label:"Aerial",   emoji:"🛰️"},
  {id:"medical", label:"Medical",  emoji:"🩻"},
  {id:"wildlife",label:"Wildlife", emoji:"🦁"},
  {id:"factory", label:"Factory",  emoji:"🏭"},
];

const STEPS = ["Task & Model","Dataset","Augmentation","Hyperparameters","Generate"];

// ═══════════════════════════════════════════════════════════════════════════
// CANVAS SCENE RENDERER
// ═══════════════════════════════════════════════════════════════════════════
function drawBox(ctx, x, y, w, h, color, label) {
  const fs = Math.max(9, w * 0.04);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.strokeRect(x,y,w,h);
  const cs = 7;
  ctx.lineWidth = 3;
  [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([cx,cy])=>{
    const sx=cx===x?1:-1, sy=cy===y?1:-1;
    ctx.beginPath(); ctx.moveTo(cx,cy+sy*cs); ctx.lineTo(cx,cy); ctx.lineTo(cx+sx*cs,cy); ctx.stroke();
  });
  ctx.font=`bold ${fs}px JetBrains Mono,monospace`;
  const tw=ctx.measureText(label).width+8;
  ctx.fillStyle=color; ctx.fillRect(x,y-fs-4,tw,fs+4);
  ctx.fillStyle="#000"; ctx.fillText(label,x+4,y-3);
}

function drawScene(ctx, id, W, H) {
  ctx.clearRect(0,0,W,H);
  if (id==="street") {
    const sky=ctx.createLinearGradient(0,0,0,H*.55);
    sky.addColorStop(0,"#0f1f3d"); sky.addColorStop(1,"#1a4a7a");
    ctx.fillStyle=sky; ctx.fillRect(0,0,W,H*.55);
    ctx.fillStyle="#1e1e1e"; ctx.fillRect(0,H*.55,W,H*.45);
    ctx.strokeStyle="#f5c518"; ctx.lineWidth=2; ctx.setLineDash([16,12]);
    ctx.beginPath(); ctx.moveTo(W*.5,H*.55); ctx.lineTo(W*.5,H); ctx.stroke(); ctx.setLineDash([]);
    [[.04,.12,.13,.48,"#0e2a4a"],[.18,.05,.15,.6,"#0a1f38"],[.36,.14,.11,.5,"#112233"],[.63,.08,.14,.58,"#0e2a4a"],[.78,.16,.12,.5,"#0a1f38"],[.9,.05,.09,.6,"#081524"]].forEach(([x,y,w,h,c])=>{
      ctx.fillStyle=c; ctx.fillRect(W*x,H*y,W*w,H*h);
      ctx.fillStyle="rgba(255,240,80,.5)";
      for(let wy=H*(y+.04);wy<H*(y+h-.04);wy+=H*.065)
        for(let wx=W*(x+.02);wx<W*(x+w-.03);wx+=W*.035)
          if(Math.random()>.3) ctx.fillRect(wx,wy,W*.018,H*.04);
    });
    ctx.fillStyle="#b03020"; ctx.beginPath(); ctx.roundRect(W*.10,H*.63,W*.22,H*.14,4); ctx.fill();
    ctx.fillStyle="#111827"; ctx.fillRect(W*.13,H*.65,W*.07,H*.08); ctx.fillRect(W*.21,H*.65,W*.07,H*.08);
    ctx.fillStyle="#111"; [W*.15,W*.30].forEach(x=>{ctx.beginPath();ctx.arc(x,H*.77,W*.022,0,Math.PI*2);ctx.fill();});
    ctx.fillStyle="#607080"; ctx.beginPath(); ctx.roundRect(W*.60,H*.62,W*.25,H*.14,4); ctx.fill();
    ctx.fillStyle="#111827"; ctx.fillRect(W*.63,H*.64,W*.08,H*.08); ctx.fillRect(W*.72,H*.64,W*.08,H*.08);
    [W*.655,W*.815].forEach(x=>{ctx.fillStyle="#111";ctx.beginPath();ctx.arc(x,H*.76,W*.022,0,Math.PI*2);ctx.fill();});
    ctx.fillStyle="#d4854a"; ctx.fillRect(W*.455,H*.50,W*.05,H*.18);
    ctx.fillStyle="#e0a060"; ctx.beginPath(); ctx.arc(W*.48,H*.485,W*.026,0,Math.PI*2); ctx.fill();
    ctx.fillStyle="#a03020"; ctx.fillRect(W*.45,H*.56,W*.02,H*.11); ctx.fillRect(W*.485,H*.56,W*.02,H*.11);
    drawBox(ctx,W*.09,H*.60,W*.24,H*.19,"#00ff41","car 0.94");
    drawBox(ctx,W*.59,H*.59,W*.27,H*.19,"#00ff41","car 0.91");
    drawBox(ctx,W*.445,H*.46,W*.065,H*.24,"#ff4444","person 0.88");
  }
  else if (id==="aerial") {
    const g=ctx.createLinearGradient(0,0,W,H);
    g.addColorStop(0,"#1e3a10"); g.addColorStop(.5,"#2d5a1e"); g.addColorStop(1,"#142810");
    ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle="rgba(80,160,40,.25)"; ctx.lineWidth=1;
    for(let i=0;i<=8;i++){ctx.beginPath();ctx.moveTo(W*i/8,0);ctx.lineTo(W*i/8,H);ctx.stroke();}
    for(let i=0;i<=6;i++){ctx.beginPath();ctx.moveTo(0,H*i/6);ctx.lineTo(W,H*i/6);ctx.stroke();}
    ctx.strokeStyle="#7a6040"; ctx.lineWidth=7;
    ctx.beginPath(); ctx.moveTo(0,H*.45); ctx.lineTo(W,H*.45); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W*.5,0); ctx.lineTo(W*.5,H); ctx.stroke();
    [[.06,.06,.16,.14,"#4a3a28"],[.26,.04,.13,.16,"#3a2a18"],[.66,.08,.18,.14,"#4a3a28"],[.06,.58,.15,.16,"#3a2a18"],[.54,.60,.17,.15,"#4a3a28"],[.74,.56,.13,.17,"#3a2a18"]].forEach(([x,y,w,h,c])=>{
      ctx.fillStyle="rgba(0,0,0,.35)"; ctx.fillRect(W*(x+.01),H*(y+.01),W*w,H*h);
      ctx.fillStyle=c; ctx.fillRect(W*x,H*y,W*w,H*h);
    });
    [[.16,.42,"#e74c3c"],[.31,.43,"#3498db"],[.61,.43,"#e67e22"],[.76,.44,"#2ecc71"]].forEach(([x,y,c])=>{
      ctx.fillStyle=c; ctx.fillRect(W*x,H*y,W*.042,H*.022);
    });
    ctx.fillStyle="rgba(20,80,180,.55)"; ctx.beginPath();
    ctx.ellipse(W*.86,H*.28,W*.09,H*.13,.3,0,Math.PI*2); ctx.fill();
    [[.06,.05,.16,.15,"building"],[.26,.03,.13,.17,"building"],[.66,.07,.18,.15,"building"],[.06,.57,.15,.17,"building"],[.54,.59,.17,.16,"building"]].forEach(([x,y,w,h,l])=>{
      drawBox(ctx,W*x,H*y,W*w,H*h,"#ffd700",l+" 0.9"+Math.floor(Math.random()*9));
    });
    [[.16,.415,.042,.024,"car"],[.31,.424,.042,.024,"car"],[.61,.424,.042,.024,"car"]].forEach(([x,y,w,h,l])=>{
      drawBox(ctx,W*x,H*y,W*w,H*h,"#00ff41",l+" 0.8"+Math.floor(Math.random()*9));
    });
  }
  else if (id==="medical") {
    ctx.fillStyle="#030810"; ctx.fillRect(0,0,W,H);
    const cx=W*.5,cy=H*.5;
    const b=ctx.createRadialGradient(cx,cy,W*.08,cx,cy,W*.43);
    b.addColorStop(0,"rgba(190,200,220,.14)"); b.addColorStop(.7,"rgba(140,155,180,.07)"); b.addColorStop(1,"rgba(0,0,0,0)");
    ctx.fillStyle=b; ctx.beginPath(); ctx.ellipse(cx,cy,W*.42,H*.48,0,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle="rgba(190,205,230,.45)"; ctx.lineWidth=1.5;
    for(let i=0;i<7;i++){
      const y2=H*.27+i*H*.066;
      ctx.beginPath(); ctx.ellipse(cx,y2,W*.21,H*.038,0,0,Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(cx,y2,W*.21,H*.038,0,Math.PI,Math.PI*2); ctx.stroke();
    }
    ctx.fillStyle="rgba(215,220,240,.38)";
    for(let i=0;i<9;i++) ctx.fillRect(W*.484,H*.21+i*H*.056,W*.032,H*.04);
    const hg=ctx.createRadialGradient(W*.43,H*.42,0,W*.43,H*.42,W*.09);
    hg.addColorStop(0,"rgba(210,45,45,.65)"); hg.addColorStop(1,"rgba(140,20,20,.05)");
    ctx.fillStyle=hg; ctx.beginPath(); ctx.ellipse(W*.43,H*.44,W*.09,H*.11,0,0,Math.PI*2); ctx.fill();
    const lg=ctx.createRadialGradient(W*.32,H*.44,0,W*.32,H*.44,W*.14);
    lg.addColorStop(0,"rgba(170,185,210,.28)"); lg.addColorStop(1,"rgba(0,0,0,0)");
    ctx.fillStyle=lg; ctx.beginPath(); ctx.ellipse(W*.31,H*.44,W*.13,H*.18,0,0,Math.PI*2); ctx.fill();
    const rg=ctx.createRadialGradient(W*.63,H*.44,0,W*.63,H*.44,W*.14);
    rg.addColorStop(0,"rgba(170,185,210,.28)"); rg.addColorStop(1,"rgba(0,0,0,0)");
    ctx.fillStyle=rg; ctx.beginPath(); ctx.ellipse(W*.64,H*.44,W*.13,H*.18,0,0,Math.PI*2); ctx.fill();
    drawBox(ctx,W*.34,H*.32,W*.18,H*.24,"#00ff41","heart 0.96");
    drawBox(ctx,W*.17,H*.27,W*.27,H*.36,"#00ccff","left_lung 0.94");
    drawBox(ctx,W*.50,H*.27,W*.27,H*.36,"#00ccff","right_lung 0.93");
  }
  else if (id==="wildlife") {
    const sky=ctx.createLinearGradient(0,0,0,H*.6);
    sky.addColorStop(0,"#e8521a"); sky.addColorStop(.38,"#f9b630"); sky.addColorStop(1,"#7ab04a");
    ctx.fillStyle=sky; ctx.fillRect(0,0,W,H*.6);
    const gnd=ctx.createLinearGradient(0,H*.6,0,H);
    gnd.addColorStop(0,"#c09840"); gnd.addColorStop(1,"#7a5510");
    ctx.fillStyle=gnd; ctx.fillRect(0,H*.6,W,H*.4);
    ctx.fillStyle="#3a2510"; ctx.fillRect(W*.73,H*.24,W*.022,H*.38);
    ctx.fillStyle="#254820"; ctx.beginPath();
    ctx.ellipse(W*.74,H*.21,W*.11,H*.07,-.2,0,Math.PI*2); ctx.fill();
    ctx.fillStyle="#bd8c35"; ctx.beginPath();
    ctx.ellipse(W*.26,H*.63,W*.18,H*.10,0,0,Math.PI*2); ctx.fill();
    const mane=ctx.createRadialGradient(W*.41,H*.58,0,W*.41,H*.58,W*.09);
    mane.addColorStop(0,"#c89040"); mane.addColorStop(.5,"#7a3810"); mane.addColorStop(1,"rgba(0,0,0,0)");
    ctx.fillStyle=mane; ctx.beginPath(); ctx.arc(W*.41,H*.575,W*.09,0,Math.PI*2); ctx.fill();
    ctx.fillStyle="#d0a040"; ctx.beginPath(); ctx.arc(W*.41,H*.57,W*.065,0,Math.PI*2); ctx.fill();
    ctx.fillStyle="#1a1000"; [W*.395,W*.425].forEach(x=>{ctx.beginPath();ctx.arc(x,H*.558,W*.008,0,Math.PI*2);ctx.fill();});
    ctx.fillStyle="#d8d8d0"; ctx.beginPath(); ctx.ellipse(W*.62,H*.665,W*.12,H*.08,0,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(W*.724,H*.625,W*.045,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle="#111"; ctx.lineWidth=2;
    for(let i=0;i<5;i++){ctx.beginPath();ctx.moveTo(W*(.52+i*.024),H*.598);ctx.lineTo(W*(.52+i*.024),H*.722);ctx.stroke();}
    ctx.fillStyle="#787878"; ctx.beginPath(); ctx.ellipse(W*.12,H*.665,W*.09,H*.10,0,0,Math.PI*2); ctx.fill();
    ctx.fillStyle="#686868"; ctx.beginPath(); ctx.arc(W*.20,H*.615,W*.058,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle="#787878"; ctx.lineWidth=5;
    ctx.beginPath(); ctx.moveTo(W*.225,H*.638); ctx.quadraticCurveTo(W*.26,H*.675,W*.225,H*.715); ctx.stroke();
    drawBox(ctx,W*.025,H*.51,W*.18,H*.21,"#ff6b35","elephant 0.97");
    drawBox(ctx,W*.08,H*.525,W*.36,H*.18,"#ff3838","lion 0.95");
    drawBox(ctx,W*.48,H*.555,W*.28,H*.19,"#00ff41","zebra 0.92");
  }
  else if (id==="factory") {
    ctx.fillStyle="#0e0e12"; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle="rgba(50,50,70,.55)"; ctx.lineWidth=1;
    for(let i=0;i<=10;i++){ctx.beginPath();ctx.moveTo(W*i/10,0);ctx.lineTo(W*i/10,H);ctx.stroke();}
    for(let i=0;i<=8;i++){ctx.beginPath();ctx.moveTo(0,H*i/8);ctx.lineTo(W,H*i/8);ctx.stroke();}
    ctx.fillStyle="#222228"; ctx.fillRect(W*.04,H*.54,W*.92,H*.13);
    ctx.strokeStyle="#404050"; ctx.lineWidth=1.5;
    for(let i=0;i<13;i++){ctx.fillStyle="#2a2a32";ctx.fillRect(W*(.06+i*.073),H*.55,W*.038,H*.11);ctx.strokeRect(W*(.06+i*.073),H*.55,W*.038,H*.11);}
    [[.09,.555,"#1a5a28"],[.235,.555,"#1a5a28"],[.455,.555,"#1a5a28"],[.71,.555,"#5a1818"]].forEach(([x,y,c])=>{
      ctx.fillStyle=c; ctx.fillRect(W*x,H*y,W*.10,H*.083);
      ctx.fillStyle="rgba(180,180,180,.35)";
      ctx.fillRect(W*(x+.01),H*(y+.01),W*.022,H*.016);
      ctx.fillRect(W*(x+.04),H*(y+.01),W*.022,H*.016);
      ctx.fillRect(W*(x+.02),H*(y+.042),W*.06,H*.009);
    });
    ctx.strokeStyle="#4a9eff"; ctx.lineWidth=7;
    ctx.beginPath(); ctx.moveTo(W*.5,H*.09); ctx.lineTo(W*.455,H*.30); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W*.455,H*.30); ctx.lineTo(W*.48,H*.535); ctx.stroke();
    [W*.5,W*.455].forEach((x,i)=>{const y=i===0?H*.09:H*.30; ctx.fillStyle="#4a9eff";ctx.beginPath();ctx.arc(x,y,W*.018,0,Math.PI*2);ctx.fill();});
    ctx.fillStyle="rgba(255,40,40,.25)"; ctx.fillRect(W*.71,H*.555,W*.10,H*.083);
    ctx.strokeStyle="rgba(255,60,60,.7)"; ctx.lineWidth=2; ctx.strokeRect(W*.71,H*.555,W*.10,H*.083);
    ctx.fillStyle="#262636"; ctx.fillRect(W*.01,H*.09,W*.17,H*.41);
    ctx.fillStyle="#3a4a5e"; ctx.fillRect(W*.04,H*.13,W*.11,H*.09);
    ctx.fillStyle="#4a9eff"; ctx.beginPath(); ctx.arc(W*.095,H*.30,W*.028,0,Math.PI*2); ctx.fill();
    ctx.fillStyle="#262636"; ctx.fillRect(W*.82,H*.09,W*.17,H*.41);
    ctx.fillStyle="#4a9eff"; ctx.beginPath(); ctx.arc(W*.905,H*.30,W*.028,0,Math.PI*2); ctx.fill();
    drawBox(ctx,W*.085,H*.542,W*.115,H*.107,"#00ff41","pcb_ok 0.99");
    drawBox(ctx,W*.228,H*.542,W*.115,H*.107,"#00ff41","pcb_ok 0.97");
    drawBox(ctx,W*.448,H*.542,W*.115,H*.107,"#00ff41","pcb_ok 0.98");
    drawBox(ctx,W*.706,H*.538,W*.123,H*.113,"#ff3838","defect 0.94");
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// AUGMENTATION ENGINE (pure canvas math)
// ═══════════════════════════════════════════════════════════════════════════
function applyAugs(srcCanvas, params) {
  const W=srcCanvas.width, H=srcCanvas.height;
  const mk=()=>{const c=document.createElement("canvas");c.width=W;c.height=H;return c;};
  let out=mk(); out.getContext("2d").drawImage(srcCanvas,0,0);

  const swap=(c)=>{out=c;};

  // Flip LR
  if((params.fliplr||0)>0.5){const t=mk();const tc=t.getContext("2d");tc.save();tc.scale(-1,1);tc.drawImage(out,-W,0);tc.restore();swap(t);}
  // Flip UD
  if((params.flipud||0)>0.5){const t=mk();const tc=t.getContext("2d");tc.save();tc.scale(1,-1);tc.drawImage(out,0,-H);tc.restore();swap(t);}
  // Rotation
  if((params.degrees||0)!==0){const a=(params.degrees*Math.PI)/180;const t=mk();const tc=t.getContext("2d");tc.save();tc.translate(W/2,H/2);tc.rotate(a);tc.drawImage(out,-W/2,-H/2);tc.restore();swap(t);}
  // Scale
  if((params.scale||0)>0){const s=1+(params.scale*0.4);const t=mk();const tc=t.getContext("2d");const nw=W*s,nh=H*s,ox=(W-nw)/2,oy=(H-nh)/2;tc.drawImage(out,ox,oy,nw,nh);swap(t);}
  // Translation
  if((params.translate||0)>0){const tx=W*(params.translate*0.4),ty=H*(params.translate*0.25);const t=mk();const tc=t.getContext("2d");tc.drawImage(out,tx,ty);swap(t);}
  // Shear
  if((params.shear||0)>0){const sh=params.shear/60;const t=mk();const tc=t.getContext("2d");tc.save();tc.transform(1,0,sh,1,0,0);tc.drawImage(out,0,0);tc.restore();swap(t);}
  // Perspective
  if((params.perspective||0)>0){const sk=params.perspective*60000;const t=mk();const tc=t.getContext("2d");tc.save();tc.transform(1,sk*.000008,sk*.000008,1,0,0);tc.drawImage(out,0,0);tc.restore();swap(t);}

  // HSV pixel ops
  const hShift=(params.hsv_h||0)*340;
  const sScale=1+((params.hsv_s||0.7)-0.7)*1.8;
  const vScale=1+((params.hsv_v||0.4)-0.4)*1.8;
  if(Math.abs(hShift)>2||Math.abs(sScale-1)>0.05||Math.abs(vScale-1)>0.05){
    const ctx2=out.getContext("2d");
    const id=ctx2.getImageData(0,0,W,H); const d=id.data;
    for(let i=0;i<d.length;i+=4){
      let r=d[i]/255,g=d[i+1]/255,b=d[i+2]/255;
      const mx=Math.max(r,g,b),mn=Math.min(r,g,b),df=mx-mn;
      let h=0,s=mx===0?0:df/mx,v=mx;
      if(df>0){if(mx===r)h=((g-b)/df)%6;else if(mx===g)h=(b-r)/df+2;else h=(r-g)/df+4;h*=60;if(h<0)h+=360;}
      h=(h+hShift)%360;if(h<0)h+=360;
      s=Math.min(1,Math.max(0,s*sScale));v=Math.min(1,Math.max(0,v*vScale));
      const c2=v*s,x2=c2*(1-Math.abs((h/60)%2-1)),m2=v-c2;
      let rr,gg,bb;
      if(h<60){rr=c2;gg=x2;bb=0;}else if(h<120){rr=x2;gg=c2;bb=0;}else if(h<180){rr=0;gg=c2;bb=x2;}
      else if(h<240){rr=0;gg=x2;bb=c2;}else if(h<300){rr=x2;gg=0;bb=c2;}else{rr=c2;gg=0;bb=x2;}
      d[i]=Math.round((rr+m2)*255);d[i+1]=Math.round((gg+m2)*255);d[i+2]=Math.round((bb+m2)*255);
    }
    ctx2.putImageData(id,0,0);
  }

  // Mosaic
  if((params.mosaic||0)>0.5){
    const t=mk();const tc=t.getContext("2d");
    tc.drawImage(out,0,0,W/2,H/2);
    tc.save();tc.scale(-1,1);tc.drawImage(out,-W,0,W/2,H/2);tc.restore();
    tc.save();tc.scale(1,-1);tc.drawImage(out,0,-H,W/2,H/2);tc.restore();
    tc.save();tc.scale(-1,-1);tc.drawImage(out,-W,-H,W/2,H/2);tc.restore();
    tc.strokeStyle="rgba(255,210,0,.85)";tc.lineWidth=2;
    tc.beginPath();tc.moveTo(W/2,0);tc.lineTo(W/2,H);tc.stroke();
    tc.beginPath();tc.moveTo(0,H/2);tc.lineTo(W,H/2);tc.stroke();
    swap(t);
  }

  // MixUp
  if((params.mixup||0)>0){
    const ctx2=out.getContext("2d");
    ctx2.globalAlpha=params.mixup*0.38;
    ctx2.filter="hue-rotate(100deg) saturate(0.75)";
    ctx2.drawImage(out,W*.04,H*.04,W*.95,H*.95);
    ctx2.filter="none";ctx2.globalAlpha=1;
  }

  // Random Erasing
  if((params.erasing||0)>0){
    const ctx2=out.getContext("2d");
    const n=Math.ceil(params.erasing*4);
    ctx2.fillStyle="rgba(15,18,24,1)";
    for(let i=0;i<n;i++){
      const rx=W*.08+Math.random()*W*.48,ry=H*.08+Math.random()*H*.48;
      const rw=W*.07+Math.random()*W*.18,rh=H*.05+Math.random()*H*.15;
      ctx2.fillRect(rx,ry,rw,rh);
    }
  }

  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// AUGMENTATION PREVIEW COMPONENT
// ═══════════════════════════════════════════════════════════════════════════
function AugPreview({ augVals, onAugChange }) {
  const isMobile = useIsMobile();
  const [scene, setScene] = useState("street");
  const [grp, setGrp] = useState("🎨 Color");
  const srcRef = useRef(null);
  const augRef = useRef(null);
  const rafRef = useRef(null);

  const W = 420, H = 280;

  useEffect(() => {
    const c = srcRef.current;
    if (!c) return;
    c.width = W; c.height = H;
    drawScene(c.getContext("2d"), scene, W, H);
  }, [scene]);

  useEffect(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const src = srcRef.current, aug = augRef.current;
      if (!src || !aug) return;
      aug.width = W; aug.height = H;
      const tmp = document.createElement("canvas"); tmp.width = W; tmp.height = H;
      drawScene(tmp.getContext("2d"), scene, W, H);
      const result = applyAugs(tmp, augVals || {});
      const ac = aug.getContext("2d");
      ac.clearRect(0, 0, W, H); ac.drawImage(result, 0, 0, W, H);
      ac.fillStyle = "rgba(0,0,0,0.4)"; ac.fillRect(0, H - 22, W, 22);
      ac.fillStyle = "#fff"; ac.font = `bold 11px var(--font-mono)`;
      const active = Object.entries(augVals || {}).filter(([, v]) => v > 0).length;
      ac.fillText(`▶ AUGMENTED (${active} active)`, 12, H - 7);
    });
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [augVals, scene]);

  const allItems = AUG_GROUPS.flatMap(g => g.items);
  const gc = AUG_GROUPS.find(g => g.title === grp)?.gc || "var(--accent-primary)";
  const groupItems = AUG_GROUPS.find(g => g.title === grp)?.items || [];

  const reset = () => { const d = {}; allItems.forEach(p => { d[p.k] = p.def; }); onAugChange?.(d); };
  const rndm = () => { const d = {}; allItems.forEach(p => { d[p.k] = parseFloat((p.min + Math.random() * (p.max - p.min)).toFixed(5)); }); onAugChange?.(d); };

  return (
    <div style={{color: "var(--text-primary)"}}>
      <div className="aug-header">
        <div className="hd" style={{marginBottom: 0}}>Vision Simulator</div>
        <div className="aug-btns">
          <button onClick={reset} className="btn btn-g" style={{padding: "6px 14px", fontSize: 12}}>↺ Default</button>
          <button onClick={rndm} className="btn btn-p" style={{padding: "6px 14px", fontSize: 12}}>⚡ Randomize</button>
        </div>
      </div>

      <div className="ftabs" style={{marginBottom: 20}}>
        {SCENES.map(s => (
          <div key={s.id} onClick={() => setScene(s.id)}
            className={`ftab${scene === s.id ? " sel" : ""}`}
            style={scene === s.id ? {color: "var(--accent-primary)", borderColor: "var(--accent-primary)"} : {}}>
            {s.emoji} {s.label}
          </div>
        ))}
      </div>

      <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: isMobile ? 10 : 20, marginBottom: 24}}>
        {[{label: "Source Frame", color: "var(--text-tertiary)", ref: srcRef}, {label: "Augmented Result", color: "var(--accent-primary)", ref: augRef}].map(({label, color, ref}) => (
          <div key={label}>
            <div style={{fontSize: 11, fontWeight: 700, color, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.5px", display: "flex", alignItems: "center", gap: 6}}>
              <span style={{width: 6, height: 6, background: color, borderRadius: "50%"}} />
              {label}
            </div>
            <div style={{borderRadius: 16, overflow: "hidden", border: `1px solid var(--border-color)`, boxShadow: "var(--shadow-sm)"}}>
              <canvas ref={ref} style={{display: "block", width: "100%", height: "auto"}} />
            </div>
          </div>
        ))}
      </div>

      <div className="gtabs" style={{marginBottom: 24, borderRadius: 10, background: "var(--bg-tertiary)"}}>
        {AUG_GROUPS.map(g => (
          <div key={g.title} onClick={() => setGrp(g.title)}
            className={`gtab${grp === g.title ? " sel" : ""}`}
            style={grp === g.title ? {color: g.gc, borderBottomColor: g.gc} : {}}>
            {g.title}
          </div>
        ))}
      </div>

      <div style={{display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 24, marginBottom: 32}}>
        {groupItems.map(p => {
          const val = augVals?.[p.k] ?? p.def;
          const pct = ((val - p.min) / (p.max - p.min)) * 100;
          const changed = Math.abs(val - p.def) > 0.001;
          return (
            <div key={p.k}>
              <div style={{display: "flex", justifyContent: "space-between", marginBottom: 10}}>
                <span style={{fontSize: 13, fontWeight: 600, color: "var(--text-secondary)"}}>{p.l}</span>
                <span style={{fontSize: 13, fontWeight: 700, color: changed ? "var(--accent-primary)" : "var(--text-tertiary)", fontFamily: "var(--font-mono)"}}>
                  {Number(val).toFixed(p.step < 0.01 ? 4 : 2)}
                </span>
              </div>
              <div style={{position: "relative", height: 20, display: "flex", alignItems: "center", marginBottom: 6}}>
                <div style={{position: "absolute", left: 0, right: 0, height: 4, background: "var(--bg-tertiary)", borderRadius: 2}} />
                <div style={{position: "absolute", left: 0, width: `${pct}%`, height: 4, background: changed ? "var(--accent-primary)" : "var(--text-tertiary)", borderRadius: 2}} />
                <input type="range" min={p.min} max={p.max} step={p.step} value={val}
                  onChange={e => onAugChange?.({...augVals, [p.k]: parseFloat(e.target.value)})}
                  style={{position: "absolute", inset: 0, width: "100%", opacity: 0, height: 20, cursor: "pointer", margin: 0, zIndex: 2}} />
                <div style={{position: "absolute", left: `calc(${pct}% - 8px)`, width: 16, height: 16, background: "#fff", border: `2px solid ${changed ? "var(--accent-primary)" : "var(--border-color)"}`, borderRadius: "50%", boxShadow: "var(--shadow-sm)", zIndex: 1, pointerEvents: "none"}} />
              </div>
              <div style={{fontSize: 11, color: "var(--text-tertiary)"}}>{p.tip}</div>
            </div>
          );
        })}
      </div>

      <div className="ctags">
        {allItems.map(p => {
          const val = augVals?.[p.k] ?? p.def;
          const on = Math.abs(val - p.def) > 0.001;
          return (
            <div key={p.k} className="ctag" style={{opacity: on ? 1 : 0.4, background: on ? "rgba(217, 70, 168, 0.05)" : "transparent", borderColor: on ? "var(--accent-primary)" : "var(--border-color)"}}>
               <span style={{color: on ? "var(--accent-primary)" : "inherit"}}>{p.l}</span>
               {on && <span className="sv" style={{fontSize: 10, marginTop: 0}}>{Number(val).toFixed(1)}</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL DROPDOWN
// ═══════════════════════════════════════════════════════════════════════════
function ModelDropdown({ taskId, value, onChange }) {
  const isMobile = useIsMobile();
  const [open, setOpen] = useState(false);
  const [srch, setSrch] = useState("");
  const [fam, setFam] = useState("all");
  const fams = ["all", "v5", "v8", "v9", "v10", "v11", "v26", "rtdetr", "world", "nas"];
  const models = ALL_MODELS.filter(m => m.t.includes(taskId) && (fam === "all" || m.f === fam) && (!srch || m.n.toLowerCase().includes(srch.toLowerCase())));
  const sel = ALL_MODELS.find(m => m.id === value);

  const dropdownStyle = isMobile ? {
    position: "fixed", top: 0, left: 0, right: 0, bottom: 0, zIndex: 9999,
    background: "#fff", display: "flex", flexDirection: "column", overflow: "hidden",
    borderRadius: 0, boxShadow: "none", border: "none", maxHeight: "100dvh",
  } : {
    position: "absolute", top: "calc(100% + 12px)", left: 0, right: 0, zIndex: 1000,
    background: "#fff", border: "1px solid var(--border-color)", borderRadius: 16,
    boxShadow: "var(--shadow-lg)", maxHeight: 520, display: "flex", flexDirection: "column", overflow: "hidden",
  };

  return (
    <div style={{position: "relative"}}>
      <button onClick={() => setOpen(o => !o)} 
        style={{width: "100%", padding: isMobile ? "14px 16px" : "16px 20px", background: "#fff", border: `1px solid ${open ? "var(--accent-primary)" : "var(--border-color)"}`, borderRadius: 12, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, transition: "all .2s", boxShadow: open ? "0 0 0 4px rgba(217, 70, 168, 0.1)" : "var(--shadow-sm)"}}>
        <div style={{display: "flex", alignItems: "center", gap: 10, overflow: "hidden"}}>
           <div style={{width: 32, height: 32, flexShrink: 0, background: sel ? "var(--accent-primary)" : "var(--bg-tertiary)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 13, fontWeight: 800}}>
             {sel ? sel.n[0].toUpperCase() : "?"}
           </div>
           <span style={{fontWeight: 700, color: sel ? "var(--text-primary)" : "var(--text-tertiary)", fontSize: isMobile ? 14 : 16, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis"}}>
             {sel ? `${sel.n} — ${sel.p}` : "Select Architecture..."}
           </span>
        </div>
        <span style={{color: "var(--text-tertiary)", transform: open ? "rotate(180deg)" : "none", transition: "transform .2s", flexShrink: 0}}>▼</span>
      </button>
      {open && (
        <div style={dropdownStyle}>
          {isMobile && (
            <div style={{display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 16px 0", borderBottom: "1px solid var(--border-color)", paddingBottom: 12}}>
              <span style={{fontWeight: 700, fontSize: 16, color: "var(--text-primary)"}}>Select Architecture</span>
              <button onClick={() => { setOpen(false); setSrch(""); }} style={{background: "var(--bg-secondary)", border: "none", borderRadius: 8, padding: "6px 14px", fontWeight: 700, cursor: "pointer", fontSize: 14}}>✕ Close</button>
            </div>
          )}
          <div style={{padding: isMobile ? "12px 16px" : 16, borderBottom: "1px solid var(--border-color)", background: "var(--bg-secondary)", flexShrink: 0}}>
            <input autoFocus placeholder="Search architectures..." value={srch} onChange={e => setSrch(e.target.value)}
              style={{width: "100%", background: "#fff", border: "1px solid var(--border-color)", borderRadius: 10, padding: "12px 16px", fontSize: 14, outline: "none"}} />
          </div>
          <div className="ftabs" style={{padding: "10px 16px", borderBottom: "1px solid var(--border-color)", marginBottom: 0, background: "var(--bg-tertiary)", overflowX: "auto", flexShrink: 0, flexWrap: "nowrap", whiteSpace: "nowrap"}}>
            {fams.map(f => <div key={f} onClick={() => setFam(f)} className={`ftab${fam === f ? " sel" : ""}`} style={{padding: "6px 12px", fontSize: 10, flexShrink: 0, display: "inline-block"}}>{f.toUpperCase()}</div>)}
          </div>
          <div style={{overflowY: "auto", flex: 1, minHeight: 0, WebkitOverflowScrolling: "touch"}}>
            {models.length === 0 && <div style={{padding: 32, textAlign: "center", color: "var(--text-tertiary)", fontSize: 14}}>No matching models.</div>}
            {models.map(m => {
              const sc = {"ultralytics": "#D946A8", "tencent": "#A855C7", "deci": "#10B981"}[m.src] || "#D946A8";
              return (
                <div key={m.id} onClick={() => { onChange(m); setOpen(false); setSrch(""); }}
                  style={{display: "flex", alignItems: "center", gap: 16, padding: isMobile ? "16px" : "14px 20px", cursor: "pointer", transition: "background .15s", borderBottom: "1px solid var(--bg-secondary)", background: m.id === value ? "rgba(217, 70, 168, 0.05)" : "transparent"}}>
                  <div style={{flex: 1, minWidth: 0}}>
                    <div style={{fontWeight: 700, color: "var(--text-primary)", fontSize: isMobile ? 14 : 15, marginBottom: 2}}>{m.n}</div>
                    <div style={{fontSize: 12, color: "var(--text-tertiary)"}}>{m.p} · mAP {m.m}</div>
                  </div>
                  <div style={{textAlign: "right", flexShrink: 0}}>
                    <div className="tag" style={{background: `${sc}15`, color: sc, border: "none", fontWeight: 700}}>{m.src}</div>
                    <div style={{fontSize: 10, color: "var(--text-tertiary)", marginTop: 4, fontFamily: "var(--font-mono)"}}>{m.fps} FPS</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════
const initAugs=()=>{const a={};AUG_GROUPS.forEach(g=>g.items.forEach(it=>{a[it.k]={enabled:it.defOn,value:it.def};}));return a;};
const flatAugs=augs=>{const f={};Object.entries(augs).forEach(([k,v])=>{f[k]=typeof v==="object"?v.value:v;});return f;};
const dl=(content,name,type="application/json")=>{const a=Object.assign(document.createElement("a"),{href:URL.createObjectURL(new Blob([content],{type})),download:name});a.click();URL.revokeObjectURL(a.href);};

// ═══════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════
export default function App() {
  const isMobile = useIsMobile();
  const col2 = isMobile ? "1fr" : "1fr 1fr";
  const [step,  setStep]  = useState(0);
  const [task,  setTask]  = useState(null);
  const [model, setMdl]   = useState(null);
  const [mode,  setMode]  = useState("finetune");
  const [fmt,   setFmt]   = useState("yolo");
  const [showG, setShowG] = useState(false);
  const [gTab,  setGTab]  = useState("structure");
  const [trainSplit,setTs]= useState(80);
  const [valSplit,  setVs]= useState(10);
  const [testSplit, setTe]= useState(10);
  const [imgSz, setImgSz] = useState(640);
  const [shuffle,setShuffle]=useState(true);
  const [seed,  setSeed]  = useState(42);
  const [classes,setCls]  = useState([]);
  const [clsIn, setClsIn] = useState("");
  const [augs,  setAugs]  = useState(initAugs);
  const [platform,setPlat]= useState("colab");
  const [loading,setLoad] = useState(false);
  const [result, setRes]  = useState(null);
  const [logs,   setLogs] = useState([]);
  const [outTab, setOutTab]= useState("notebook");
  const [backOk, setBack] = useState(null);
  const [hp, setHp] = useState({
    epochs:100,patience:50,batch:16,optimizer:"SGD",device:"0",workers:8,fraction:1.0,
    lr0:0.01,lrf:0.01,cos_lr:false,warmup_epochs:3.0,warmup_momentum:0.8,warmup_bias_lr:0.1,
    momentum:0.937,weight_decay:0.0005,dropout:0.0,
    box:7.5,cls:0.5,dfl:1.5,close_mosaic:10,amp:true,multi_scale:false,
    freeze:0,overlap_mask:true,mask_ratio:4,resume:false,
    val:true,plots:true,save:true,save_period:-1,
    project:"runs/train",name:"exp",exist_ok:false,cache:"false",
  });
  const sH=(k,v)=>setHp(h=>({...h,[k]:v}));

  useEffect(()=>{fetch(`${API}/health`).then(r=>setBack(r.ok)).catch(()=>setBack(false));}, []);
  useEffect(()=>{if(task){const first=ALL_MODELS.find(m=>m.t.includes(task.id));setMdl(first||null);setMode("finetune");}}, [task]);

  const flatA = flatAugs(augs);
  const augOnCount = Object.values(flatA).filter(v=>v>0).length;

  const buildCfg = ()=>({
    task:task?.id||"detect", model_id:model?.id||"yolov8n",
    model_pt:model?.pt||"yolov8n.pt", model_yaml:model?.y||"yolov8n.yaml",
    train_mode:mode, platform, data_format:fmt, image_size:imgSz,
    train_split:trainSplit/100, val_split:valSplit/100, test_split:testSplit/100,
    shuffle, seed, class_names:classes,
    augmentations:Object.fromEntries(Object.entries(augs).map(([k,v])=>[k,typeof v==="object"?{enabled:v.enabled,value:v.value}:v])),
    hp,
  });

  const handleGenerate=async()=>{
    setLoad(true);setRes(null);setLogs(["🚀 Calling LangGraph pipeline…"]);
    try {
      const r=await fetch(`${API}/generate`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({config:buildCfg()})});
      const d=await r.json(); setRes(d); setLogs(d.messages||[]);
    } catch(e){
      setLogs([`❌ Backend unreachable: ${e.message}`,"⚡ Offline mode — notebook generated client-side"]);
      setRes({success:true,notebook_json:null,data_yaml:`path: ./dataset\ntrain: images/train\nval: images/val\ntest: images/test\nnc: ${classes.length}\nnames: ${JSON.stringify(classes)}`,readme_md:`# YOLOForge\nModel: ${model?.n}\nTask: ${task?.label}`,messages:[],errors:[]});
    } finally{setLoad(false);}
  };

  const setSplitTrain=v=>{const t=Math.min(Math.max(v,50),90),r=100-t,vv=Math.round(r/2);setTs(t);setVs(vv);setTe(r-vv);};
  const addCls=()=>{const n=clsIn.trim();if(n&&!classes.includes(n))setCls(c=>[...c,n]);setClsIn("");};
  const getNotebook=()=>result?.notebook_json||"{}";

  // ── CSS ─────────────────────────────────────────────────────────────────
  const CSS = `
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg-primary: #FFFFFF;
  --bg-secondary: #FAF5F7;
  --bg-tertiary: #F5EEF2;
  --accent-primary: #D946A8;
  --accent-secondary: #A855C7;
  --text-primary: #111827;
  --text-secondary: #4B5563;
  --text-tertiary: #9CA3AF;
  --border-color: #E5E7EB;
  --border-radius-lg: 32px;
  --border-radius-md: 16px;
  --border-radius-sm: 8px;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --font-main: 'Plus Jakarta Sans', sans-serif;
  --font-serif: 'Instrument Serif', serif;
  --font-mono: 'JetBrains Mono', monospace;
}
html, body, #root { min-height: 100%; background: var(--bg-primary); }
body { font-family: var(--font-main); color: var(--text-primary); line-height: 1.6; font-size: 15px; -webkit-font-smoothing: antialiased; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 10px; }

.wrap { max-width: 1400px; margin: 0 auto; padding: 0 40px 140px; }

.hdr { background: rgba(255, 255, 255, 0.5); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border-bottom: 1px solid rgba(217, 70, 168, 0.1); padding: 0 24px; position: sticky; top: 0; z-index: 1000; }
.hdr-in { max-width: 1200px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; height: 72px; }
.logo { display: flex; align-items: center; gap: 12px; text-decoration: none; }
.logo-box { width: 38px; height: 38px; background: linear-gradient(135deg, #EC4899, #A855C7); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 1.2rem; box-shadow: 0 4px 12px rgba(217, 70, 168, 0.3); }
.logo-name { font-weight: 800; font-size: 1.4rem; letter-spacing: -0.5px; color: var(--text-primary); }
.logo-name em { color: var(--accent-primary); font-style: normal; }

.hero { text-align: center; padding: 100px 24px 60px; background: transparent; position: relative; z-index: 1; }
.badge { display: inline-block; padding: 6px 16px; background: rgba(255, 255, 255, 0.2); border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 100px; font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 24px; backdrop-filter: blur(8px); }
.hero h1 { font-size: clamp(2.5rem, 6vw, 4.5rem); font-weight: 800; letter-spacing: -2px; line-height: 1.05; margin-bottom: 24px; color: #fff; text-shadow: 0 2px 20px rgba(0,0,0,0.15); }
.hero h1 span { font-family: var(--font-serif); font-style: italic; font-weight: 400; color: #fff; }
.hero p { color: rgba(255, 255, 255, 0.85); font-size: 1.1rem; max-width: 640px; margin: 0 auto 40px; line-height: 1.7; }

.bento-grid { display: grid; gap: 24px; margin-top: 40px; }
.bento-card { background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.5); border-radius: var(--border-radius-lg); padding: 32px; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06); position: relative; overflow: hidden; }
.bento-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1); border-color: rgba(217, 70, 168, 0.3); }

.sbar { background: rgba(255, 255, 255, 0.25); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 100px; padding: 6px; display: inline-flex; margin: 0 auto 40px; }
.sit { display: flex; align-items: center; cursor: pointer; padding: 8px 20px; border-radius: 100px; font-size: 14px; font-weight: 600; color: rgba(255, 255, 255, 0.7); transition: all 0.2s; }
.sit.act { background: rgba(255, 255, 255, 0.85); color: var(--accent-primary); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.sit.done { color: rgba(255, 255, 255, 0.9); }
.sit-n { margin-right: 8px; font-family: var(--font-mono); font-size: 12px; opacity: 0.6; }

.card { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.5); border-radius: var(--border-radius-md); padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04); }
.hd { font-size: 12px; letter-spacing: 1px; text-transform: uppercase; color: var(--text-tertiary); margin-bottom: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
.hd::before { content: ''; width: 4px; height: 14px; background: var(--accent-primary); border-radius: 2px; }

.tgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.tc { border: 1px solid var(--border-color); border-radius: var(--border-radius-md); padding: 16px; cursor: pointer; transition: all 0.2s; background: var(--bg-secondary); }
.tc:hover { border-color: var(--accent-primary); background: var(--bg-primary); transform: scale(1.02); }
.tc.sel { border-color: var(--accent-primary); background: white; box-shadow: 0 0 0 4px rgba(217, 70, 168, 0.1); }
.tc-em { font-size: 2rem; margin-bottom: 12px; }
.tc-name { font-weight: 700; font-size: 1rem; margin-bottom: 6px; color: var(--text-primary); }
.tc-desc { font-size: 14px; color: var(--text-secondary); line-height: 1.5; }
.tc-uc { font-size: 12px; color: var(--accent-primary); margin-top: 10px; font-family: var(--font-mono); font-weight: 500; }

.mdet { background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: var(--border-radius-sm); padding: 20px; margin-top: 20px; }
.mdet-name { font-weight: 700; font-size: 1.1rem; color: var(--text-primary); margin-bottom: 10px; }
.mdet-stats { display: flex; gap: 20px; flex-wrap: wrap; }
.mst { font-family: var(--font-mono); font-size: 13px; color: var(--text-secondary); }
.mst em { color: var(--accent-primary); font-style: normal; font-weight: 700; margin-left: 4px; }
.mdet-ref { font-family: var(--font-mono); font-size: 12px; color: var(--text-tertiary); margin-top: 10px; }

.btn { font-family: var(--font-main); font-size: 14px; padding: 12px 24px; border-radius: 100px; border: none; cursor: pointer; transition: all 0.2s; font-weight: 600; }
.btn-p { background: linear-gradient(135deg, #EC4899, #A855C7); color: #fff; box-shadow: 0 4px 12px rgba(217, 70, 168, 0.25); }
.btn-p:hover { background: linear-gradient(135deg, #DB2777, #9333C4); transform: translateY(-2px); box-shadow: 0 6px 16px rgba(217, 70, 168, 0.35); }
.btn-g { background: var(--bg-secondary); color: var(--text-primary); border: 1px solid var(--border-color); }
.btn-g:hover { background: var(--bg-tertiary); }
.btn-gen { width: 100%; padding: 18px; font-size: 16px; font-weight: 700; background: linear-gradient(135deg, #EC4899, #A855C7); color: #fff; border-radius: var(--border-radius-md); border: none; cursor: pointer; transition: all 0.3s; box-shadow: 0 10px 20px rgba(217, 70, 168, 0.2); }
.btn-gen:hover:not(:disabled) { transform: translateY(-3px); box-shadow: 0 15px 30px rgba(217, 70, 168, 0.3); }

input[type=text], input[type=number], select { width: 100%; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 12px; color: var(--text-primary); font-family: var(--font-mono); font-size: 14px; padding: 12px 16px; outline: none; transition: all 0.2s; }
input:focus, select:focus { border-color: var(--accent-primary); background: white; box-shadow: 0 0 0 4px rgba(217, 70, 168, 0.05); }

.nb { position: fixed; bottom: 32px; left: 50%; transform: translateX(-50%); z-index: 2000; background: rgba(255, 255, 255, 0.55); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); padding: 12px 24px; border-radius: 100px; border: 1px solid rgba(255, 255, 255, 0.5); box-shadow: 0 10px 40px rgba(168, 85, 199, 0.15); width: calc(100% - 48px); max-width: 1000px; }
.nb-in { display: flex; align-items: center; justify-content: space-between; }

.code { background: var(--text-primary); border-radius: var(--border-radius-md); padding: 24px; font-family: var(--font-mono); font-size: 13px; color: #E5E7EB; white-space: pre; overflow: auto; max-height: 400px; line-height: 1.8; }

.platcard { flex: 1; padding: 13px; border: 1px solid var(--border-color); border-radius: var(--border-radius-md); cursor: pointer; text-align: center; transition: all 0.15s; background: var(--bg-secondary); }
.platcard:hover { border-color: var(--accent-primary); transform: translateY(-2px); }
.platcard.sel { border-color: var(--accent-primary); background: white; box-shadow: var(--shadow-md); }
.plat-em { font-size: 1.5rem; margin-bottom: 8px; }
.plat-n { font-weight: 700; font-size: 14px; color: var(--text-primary); }
.plat-s { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

.sgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
.sitem { background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: var(--border-radius-sm); padding: 14px; }
.sk { font-family: var(--font-mono); font-size: 11px; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.5px; }
.sv { font-family: var(--font-mono); font-size: 14px; color: var(--accent-primary); font-weight: 700; margin-top: 4px; word-break: break-all; }

.otabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.otab { padding: 8px 16px; border: 1px solid var(--border-color); border-radius: 100px; cursor: pointer; font-size: 13px; font-weight: 600; color: var(--text-secondary); transition: all .15s; }
.otab.sel { border-color: var(--accent-primary); color: var(--accent-primary); background: rgba(217, 70, 168, 0.05); }

.dlrow { display: flex; align-items: center; justify-content: space-between; padding: 16px 0; border-bottom: 1px solid var(--border-color); gap: 16px; }
.dlrow:last-child { border-bottom: none; }
.dl-n { font-family: var(--font-mono); font-size: 14px; font-weight: 600; color: var(--text-primary); }
.dl-d { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }

.logbox { background: var(--text-primary); border-radius: var(--border-radius-md); padding: 20px; font-family: var(--font-mono); font-size: 13px; color: #E5E7EB; max-height: 300px; overflow-y: auto; line-height: 1.8; }

.spin { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255, 255, 255, 0.3); border-top-color: #fff; border-radius: 50%; animation: sp .7s linear infinite; vertical-align: middle; margin-right: 8px; }
@keyframes sp { to { transform: rotate(360deg); } }

.slist { list-style: none; }
.sli { display: flex; gap: 16px; align-items: flex-start; padding: 12px 0; border-bottom: 1px solid var(--border-color); }
.sli:last-child { border-bottom: none; }
.sn { width: 28px; height: 28px; background: rgba(217, 70, 168, 0.1); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; color: var(--accent-primary); font-family: var(--font-mono); font-weight: 700; flex-shrink: 0; margin-top: 2px; }
.st2 { font-size: 14px; line-height: 1.6; color: var(--text-secondary); }
.st2 code { font-family: var(--font-mono); background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px; font-size: 13px; color: var(--accent-primary); }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: 11px; background: var(--bg-secondary); color: var(--text-secondary); border: 1px solid var(--border-color); }

@media (max-width: 1024px) {
  .wrap { padding: 0 20px 160px; }
  .hdr-in { padding: 0 4px; }
}

@media (max-width: 768px) {
  /* Layout */
  .wrap { padding: 0 14px 160px; }
  .hdr { padding: 0 14px; }
  .hdr-in { height: 56px; }
  .logo-name { font-size: 1.05rem; }
  .logo-box { width: 32px; height: 32px; font-size: 1rem; }

  /* Hero landing */
  .hero { padding: 60px 16px 40px; }
  .hero h1 { font-size: clamp(1.8rem, 8vw, 2.4rem); letter-spacing: -1px; }
  .hero p { font-size: 0.95rem; }

  /* Step nav bar — scrollable pill row */
  .sbar { max-width: calc(100vw - 28px); overflow-x: auto; -webkit-overflow-scrolling: touch; flex-wrap: nowrap; border-radius: 16px; scrollbar-width: none; }
  .sbar::-webkit-scrollbar { display: none; }
  .sit { padding: 8px 14px; font-size: 12px; white-space: nowrap; flex-shrink: 0; }
  .sit-n { display: none; }

  /* Cards & bento */
  .bento-card { padding: 18px; border-radius: 20px; }
  .card { padding: 14px; }

  /* Task grid: 2 cols on mobile */
  .tgrid { grid-template-columns: 1fr 1fr; gap: 10px; }
  .tc { padding: 12px; }
  .tc-name { font-size: 0.85rem; }
  .tc-desc { font-size: 12px; }
  .tc-uc { font-size: 11px; }

  /* Buttons */
  .btn { padding: 10px 16px; font-size: 13px; }
  .btn-gen { font-size: 13px; padding: 13px 16px; }

  /* Nav bar */
  .nb { bottom: 12px; width: calc(100% - 24px); padding: 10px 14px; border-radius: 16px; }
  .nb-in { gap: 8px; }

  /* Model stats */
  .sgrid { grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .mdet-stats { gap: 8px; }
  .sitem { padding: 10px 12px; }

  /* Format tabs */
  .ftabs { gap: 6px; flex-wrap: wrap; }
  .ftab { padding: 8px 12px; font-size: 11px; }

  /* Format guide tabs */
  .gtab { font-size: 10px; padding: 8px 2px; }
  .gbody { padding: 12px; }
  .gbody pre { font-size: 11px; padding: 10px; }

  /* Split legend */
  .spleg { flex-wrap: wrap; gap: 10px; }

  /* Classes row */
  .clsrow { flex-direction: column; gap: 8px; }
  .clsrow input { width: 100%; }
  .clsrow .btn { width: 100%; border-radius: 12px !important; text-align: center; }

  /* Download rows */
  .dlrow { flex-direction: column; align-items: flex-start; gap: 8px; }
  .dlrow .btn { width: 100%; text-align: center; border-radius: 10px !important; }

  /* Output tabs */
  .otabs { gap: 6px; }
  .otab { padding: 6px 12px; font-size: 11px; }

  /* Log & code */
  .logbox { font-size: 11px; padding: 14px; max-height: 240px; }
  .code { font-size: 10px; padding: 14px; max-height: 240px; line-height: 1.6; }

  /* Instruction list */
  .sli { gap: 10px; }
  .sn { width: 24px; height: 24px; font-size: 11px; flex-shrink: 0; }
  .st2 { font-size: 13px; }

  /* Aug preview header */
  .aug-header { flex-direction: column; align-items: flex-start; gap: 10px; }
  .aug-header .aug-btns { width: 100%; display: flex; gap: 8px; }
  .aug-header .aug-btns button { flex: 1; }

  /* Misc */
  .hd { font-size: 11px; margin-bottom: 14px; }
  .lbl { font-size: 11px; }
  .tag { font-size: 10px; }
  .sv { font-size: 13px; }
  input[type=text], input[type=number], select { font-size: 14px; padding: 10px 12px; }
}

@media (max-width: 480px) {
  .hero h1 { font-size: 1.7rem; }
  .tgrid { grid-template-columns: 1fr; }
  .bento-card { padding: 14px; border-radius: 16px; }
  .sbar { border-radius: 12px; }
  .sit { padding: 7px 10px; font-size: 11px; }
  .sgrid { grid-template-columns: 1fr 1fr; }
  .ftab { padding: 7px 10px; font-size: 10px; }
  .platcard { padding: 14px !important; }
  .plat-n { font-size: 14px !important; }
  .plat-s { font-size: 12px !important; }
}
  
/* Dataset & Custom Components */
.ftabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.ftab { padding: 10px 20px; border: 1px solid var(--border-color); border-radius: 100px; cursor: pointer; font-size: 13px; font-weight: 700; color: var(--text-secondary); transition: all .2s; background: #fff; }
.ftab.sel { border-color: var(--accent-primary); color: var(--accent-primary); box-shadow: 0 0 0 4px rgba(217, 70, 168, 0.1); }

.spbar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin: 20px 0; background: var(--bg-tertiary); }
.spleg { display: flex; gap: 20px; font-size: 12px; color: var(--text-secondary); font-weight: 600; }
.spdot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }

.clsrow { display: flex; gap: 12px; margin-bottom: 20px; }
.ctags { display: flex; flex-wrap: wrap; gap: 8px; }
.ctag { display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; font-family: var(--font-mono); font-size: 13px; font-weight: 600; color: var(--text-primary); }
.cidx { font-size: 10px; color: var(--accent-primary); opacity: 0.7; }
.crm { background: none; border: none; font-size: 16px; color: var(--text-tertiary); cursor: pointer; line-height: 1; padding: 0 2px; }
.crm:hover { color: #EF4444; }

.fguide { margin-top: 20px; background: var(--bg-secondary); border-radius: var(--border-radius-md); border: 1px solid var(--border-color); overflow: hidden; }
.gtabs { display: flex; background: var(--bg-tertiary); }
.gtab { flex: 1; text-align: center; padding: 10px; font-size: 12px; font-weight: 700; color: var(--text-tertiary); cursor: pointer; border-bottom: 2px solid transparent; }
.gtab.sel { background: #fff; color: var(--accent-primary); border-bottom-color: var(--accent-primary); }
.gbody { padding: 20px; font-size: 13px; }
.gbody pre { font-family: var(--font-mono); background: var(--text-primary); color: #fff; padding: 15px; border-radius: 8px; overflow: auto; line-height: 1.5; }

.warn { padding: 16px; background: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 12px; color: #D97706; font-size: 13px; font-weight: 500; margin-top: 16px; }
.info { padding: 16px; background: rgba(217, 70, 168, 0.05); border: 1px solid rgba(217, 70, 168, 0.2); border-radius: 12px; color: var(--accent-primary); font-size: 13px; font-weight: 500; margin-top: 16px; }
.ok { padding: 16px; background: rgba(52, 211, 153, 0.05); border: 1px solid rgba(52, 211, 153, 0.2); border-radius: 12px; color: #059669; font-size: 13px; font-weight: 500; margin-top: 16px; }

.mcard { padding: 20px; border: 1px solid var(--border-color); border-radius: 16px; cursor: pointer; transition: all .2s; background: #fff; }
.mcard.sel { border-color: var(--accent-primary); box-shadow: 0 0 0 4px rgba(217, 70, 168, 0.1); }
.mcard.dis { opacity: 0.5; cursor: not-allowed; background: var(--bg-tertiary); }
.mcard-t { font-weight: 700; margin-bottom: 4px; color: var(--text-primary); }
.mcard-s { font-size: 12px; color: var(--text-secondary); }

.chk { display: flex; align-items: center; gap: 10px; cursor: pointer; font-size: 14px; font-weight: 500; color: var(--text-secondary); margin-bottom: 8px; }
.chk input[type=checkbox] { width: 18px; height: 18px; border-radius: 4px; border: 2px solid var(--border-color); cursor: pointer; accent-color: var(--accent-primary); }

.lbl { display: block; font-size: 12px; font-weight: 800; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
.g2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.aug-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.aug-btns { display: flex; gap: 12px; }
`;

  // ── Premium Slider Helper ──────────────────────────────────────────────
  const Sl = ({ label, hint, min, max, step, vk, dec: d = 3 }) => {
    const val = hp[vk];
    const pct = ((val - min) / (max - min)) * 100;
    return (
      <div className="field" style={{marginBottom: 20}}>
        <div style={{display: "flex", justifyContent: "space-between", marginBottom: 10, alignItems: "center"}}>
          <label className="lbl" style={{marginBottom: 0}}>{label}{hint && <span className="tag" style={{marginLeft: 8}}>{hint}</span>}</label>
          <span className="sv" style={{fontSize: 13}}>{Number(val).toFixed(d)}</span>
        </div>
        <div className="rrow" style={{gap: 16}}>
          <div style={{flex: 1, position: "relative", height: 24, display: "flex", alignItems: "center"}}>
            <div style={{position: "absolute", left: 0, right: 0, height: 4, background: "var(--bg-tertiary)", borderRadius: 2}} />
            <div style={{position: "absolute", left: 0, width: `${pct}%`, height: 4, background: "var(--accent-primary)", borderRadius: 2}} />
            <input type="range" min={min} max={max} step={step} value={val}
              onChange={e => sH(vk, parseFloat(e.target.value))}
              style={{position: "absolute", inset: 0, width: "100%", opacity: 0, height: 24, cursor: "pointer", margin: 0, zIndex: 2}} />
            <div style={{position: "absolute", left: `calc(${pct}% - 8px)`, width: 16, height: 16, background: "#fff", borderRadius: "50%", border: "2px solid var(--accent-primary)", boxShadow: "var(--shadow-sm)", pointerEvents: "none", zIndex: 1}} />
          </div>
        </div>
      </div>
    );
  };

  const augFlat=flatAugs(augs);

  // ── STEP CONTENT ───────────────────────────────────────────────────────
  const stepContent=[
    // STEP 0 — Task & Model
    <div key="s0" style={{display: "grid", gridTemplateColumns: task && !isMobile ? "1fr 1fr" : "1fr", gap: isMobile ? 16 : 32, alignItems: "start"}}>
      <div>
        <div className="hd">Select Task</div>
        <div className="tgrid">
          {TASKS.map(t => (
            <div key={t.id} className={`tc${task?.id === t.id ? " sel" : ""}`} onClick={() => setTask(t)}>
              <div style={{width: "100%", height: 72, borderRadius: 8, overflow: "hidden", marginBottom: 12}}>
                <img src={t.image} alt={t.label} style={{width: "100%", height: "100%", objectFit: "cover"}} />
              </div>
              <div className="tc-name">{t.label}</div>
              <div className="tc-desc">{t.desc}</div>
              <div className="tc-uc">{t.usecase}</div>
            </div>
          ))}
        </div>
      </div>

      {task && (
        <div style={{marginTop: 0}}>
          <div className="hd">Architecture Alignment — {task.label}</div>
          <div className="card">
             <ModelDropdown taskId={task.id} value={model?.id} onChange={m => { setMdl(m); setMode("finetune"); }} />
          </div>
          
          {model && (
            <div className="mdet">
              <div style={{display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20}}>
                 <div className="mdet-name">{model.n}</div>
                 <span className="tag" style={{background:"var(--bg-tertiary)", border:"none", fontWeight:700}}>{model.f}</span>
              </div>
              <div className="mdet-stats">
                {[["Params", model.p], ["Size", model.s], ["mAP50-95", model.m], ["FPS (T4)", model.fps], ["Source", model.src]].map(([k, v]) => (
                  <div key={k} className="sitem" style={{background: "#fff", border:"none", padding: "12px 16px"}}>
                    <div className="sk" style={{fontSize: 9}}>{k}</div>
                    <div className="sv" style={{fontSize: 13}}>{v}</div>
                  </div>
                ))}
              </div>
              
              <div style={{marginTop: 32}}>
                <div className="hd">Training Strategy</div>
                <div style={{display:"grid", gridTemplateColumns: col2, gap: 16}}>
                  {[
                    ["finetune", "🎯 Fine-tune", `Recommended. Starts with ${model.pt} weights.`, true],
                    ["scratch", "⚡ From Scratch", model.y ? `Uses ${model.y} architecture.` : "Not available for this model.", !!model.y]
                  ].map(([id, lbl, sub, avail]) => (
                    <div key={id} 
                         className={`platcard${mode === id ? " sel" : ""}${!avail ? " dis" : ""}`} 
                         onClick={() => avail && setMode(id)}
                         style={{textAlign:"left", padding: 24, cursor: avail ? "pointer" : "default", opacity: avail ? 1 : 0.5}}>
                      <div className="plat-n" style={{fontSize: 16, marginBottom: 8}}>{lbl}</div>
                      <div className="plat-s" style={{fontSize: 13, lineHeight: 1.4}}>{sub}</div>
                    </div>
                  ))}
                </div>
                {mode === "scratch" && (
                  <div className="tag" style={{marginTop: 16, color: "#F59E0B", borderColor: "#F59E0B", width: "100%", padding: 12}}>
                    ⚠ Strategy Warning: Training from scratch requires significantly larger datasets and more epochs.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>,

    // STEP 1 — Dataset
    <div key="s1" style={{display: "grid", gridTemplateColumns: col2, gap: isMobile ? 16 : 32, alignItems: "start"}}>
      <div style={{display: "flex", flexDirection: "column", gap: 32}}>
        <div>
          <div className="hd">Dataset Specification</div>
          <div className="card" style={{marginBottom: 0}}>
            <div className="lbl">Annotation Format</div>
            <div className="ftabs">
              {Object.entries(FORMAT_GUIDE).map(([id, g]) => (
                <div key={id} className={`ftab${fmt === id ? " sel" : ""}`} onClick={() => setFmt(id)} style={fmt === id ? {borderColor: g.color, color: g.color} : {}}>
                  {g.label}
                </div>
              ))}
            </div>
            <button className="btn btn-g" style={{fontSize: 12, padding: "8px 16px"}} onClick={() => setShowG(o => !o)}>
              {showG ? "Hide" : "View"} Structure Guide ⎙
            </button>
            
            {showG && (
              <div className="fguide">
                <div className="gtabs">
                  {[["structure", "📁 Folder"], ["labels", "🏷 Labels"], ["yaml", "📄 YAML"], ["notes", "💡 Tips"]].map(([id, lbl]) => (
                    <div key={id} className={`gtab${gTab === id ? " sel" : ""}`} onClick={() => setGTab(id)}>{lbl}</div>
                  ))}
                </div>
                <div className="gbody">
                  {gTab === "structure" && <pre>{FORMAT_GUIDE[fmt]?.structure}</pre>}
                  {gTab === "labels" && <><pre>{FORMAT_GUIDE[fmt]?.labels}</pre>{task && <div className="tag" style={{marginTop: 10, display: "block"}}>Task Focus: {task.label_fmt}</div>}</>}
                  {gTab === "yaml" && <pre>{FORMAT_GUIDE[fmt]?.yaml}</pre>}
                  {gTab === "notes" && <div style={{fontSize: 14, lineHeight: 1.6, color: "var(--text-secondary)"}}>{FORMAT_GUIDE[fmt]?.notes}</div>}
                </div>
              </div>
            )}
          </div>
        </div>

        <div>
          <div className="hd">Ontology / Classes</div>
          <div className="card" style={{marginBottom: 0}}>
            <div className="clsrow">
              <input type="text" placeholder="Add class (e.g. 'car', 'person')..." value={clsIn}
                onChange={e => setClsIn(e.target.value)} onKeyDown={e => e.key === "Enter" && addCls()} />
              <button className="btn btn-p" onClick={addCls} style={{borderRadius: 12}}>Add</button>
            </div>
            {classes.length > 0 ? (
              <div className="ctags">
                {classes.map((c, i) => (
                  <div key={c} className="ctag">
                    <span className="cidx">{i}</span>
                    {c}
                    <button className="crm" onClick={() => setCls(n => n.filter(x => x !== c))}>×</button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="info" style={{marginTop: 0}}>
                Auto-detecting classes from dataset YAML. Add manually to override.
              </div>
            )}
          </div>
        </div>
      </div>

      <div>
        <div className="hd">Data Distribution</div>
        <div className="card" style={{marginBottom: 0}}>
          <div className="spbar">
            <div style={{background: "var(--accent-primary)", width: `${trainSplit}%`}} />
            <div style={{background: "#C084FC", width: `${valSplit}%`}} />
            <div style={{background: "#F9A8D4", width: `${testSplit}%`}} />
          </div>
          <div className="spleg">
            <span><span className="spdot" style={{background: "var(--accent-primary)"}} />Train {trainSplit}%</span>
            <span><span className="spdot" style={{background: "#C084FC"}} />Val {valSplit}%</span>
            <span><span className="spdot" style={{background: "#F9A8D4"}} />Test {testSplit}%</span>
          </div>

          <div style={{marginTop: 32}}>
             <div className="field" style={{marginBottom: 20}}>
               <div style={{display: "flex", justifyContent: "space-between", marginBottom: 10, alignItems: "center"}}>
                 <label className="lbl" style={{marginBottom: 0}}>Train Split Fraction</label>
                 <span className="sv" style={{fontSize: 13}}>{trainSplit}%</span>
               </div>
               <div className="rrow" style={{gap: 16}}>
                 <div style={{flex: 1, position: "relative", height: 24, display: "flex", alignItems: "center"}}>
                   <div style={{position: "absolute", left: 0, right: 0, height: 4, background: "var(--bg-tertiary)", borderRadius: 2}} />
                   <div style={{position: "absolute", left: 0, width: `${((trainSplit - 50) / 40) * 100}%`, height: 4, background: "var(--accent-primary)", borderRadius: 2}} />
                   <input type="range" min={50} max={90} step={5} value={trainSplit}
                     onChange={e => setSplitTrain(+e.target.value)}
                     style={{position: "absolute", inset: 0, width: "100%", opacity: 0, height: 24, cursor: "pointer", margin: 0, zIndex: 2}} />
                   <div style={{position: "absolute", left: `calc(${((trainSplit - 50) / 40) * 100}% - 8px)`, width: 16, height: 16, background: "#fff", borderRadius: "50%", border: "2px solid var(--accent-primary)", boxShadow: "var(--shadow-sm)", pointerEvents: "none", zIndex: 1}} />
                 </div>
               </div>
             </div>
             <div className="info" style={{marginTop: 0, fontSize: 12}}>
               Val / Test are automatically allocated from the remaining {100 - trainSplit}%.
             </div>
          </div>

          <div style={{display: "grid", gridTemplateColumns: col2, gap: 20, marginTop: 24}}>
            <div className="field">
              <label className="lbl">Input Resolution</label>
              <select value={imgSz} onChange={e => setImgSz(+e.target.value)}>
                {[320, 416, 512, 640, 768, 896, 1024, 1280].map(s => <option key={s} value={s}>{s}px {s === 640 ? "(Standard)" : ""}</option>)}
              </select>
            </div>
            <div className="field">
              <label className="lbl">Global Seed</label>
              <input type="number" value={seed} onChange={e => setSeed(+e.target.value)} />
            </div>
          </div>
          <label className="chk" style={{marginTop: 16}}>
            <input type="checkbox" checked={shuffle} onChange={e => setShuffle(e.target.checked)} />
            Deterministic shuffle before split
          </label>
        </div>
      </div>
    </div>,

    // STEP 2 — Augmentation (WITH LIVE PREVIEW)
    <div key="s2">
      <div className="card">
        <AugPreview
          augVals={augFlat}
          onAugChange={vals=>{
            const newAugs={...augs};
            Object.entries(vals).forEach(([k,v])=>{
              newAugs[k]={enabled:v>0,value:v};
            });
            setAugs(newAugs);
          }}
        />
      </div>
    </div>,

    // STEP 3 — Hyperparameters
    <div key="s3" style={{display: "grid", gridTemplateColumns: col2, gap: isMobile ? 16 : 32, alignItems: "start"}}>
      <div style={{display: "flex", flexDirection: "column", gap: 32}}>
        <div>
          <div className="hd">Core Parameters</div>
          <div className="card" style={{marginBottom: 0}}>
            <div style={{display: "grid", gridTemplateColumns: col2, gap: 24, marginBottom: 24}}>
              <div className="field"><label className="lbl">Training Epochs</label><input type="number" value={hp.epochs} min={1} max={1000} onChange={e => sH("epochs", +e.target.value)} /></div>
              <div className="field">
                <label className="lbl">Batch Selection</label>
                <select value={hp.batch} onChange={e => sH("batch", +e.target.value)}>
                  {[1, 2, 4, 8, 16, 32, 64, 128, -1].map(v => <option key={v} value={v}>{v === -1 ? "Dynamic (Auto)" : v}</option>)}
                </select>
              </div>
            </div>
            <div style={{display: "grid", gridTemplateColumns: col2, gap: 24}}>
              <div className="field"><label className="lbl">Early Stopping</label><input type="number" value={hp.patience} min={0} max={300} onChange={e => sH("patience", +e.target.value)} /></div>
              <div className="field">
                <label className="lbl">Computing Device</label>
                <select value={hp.device} onChange={e => sH("device", e.target.value)}>
                  <option value="0">Single GPU (0)</option>
                  <option value="cpu">CPU</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <div>
          <div className="hd">Optimization & Schedulers</div>
          <div className="card" style={{marginBottom: 0}}>
            <div style={{display: "grid", gridTemplateColumns: col2, gap: 32, marginBottom: 16}}>
              <Sl label="Learning Rate" hint="lr0" min={0.0001} max={0.1} step={0.0001} vk="lr0" dec={4} />
              <Sl label="Warmup Phases" hint="epochs" min={0} max={10} step={0.5} vk="warmup_epochs" dec={1} />
            </div>
            <label className="chk">
              <input type="checkbox" checked={hp.cos_lr} onChange={e => sH("cos_lr", e.target.checked)} />
              Enable Cosine Annealing (cos_lr)
            </label>
          </div>
        </div>
      </div>

      <div style={{display: "flex", flexDirection: "column", gap: 32}}>
        <div>
          <div className="hd">Regularization & Loss</div>
          <div className="card" style={{marginBottom: 0}}>
            <div style={{display: "grid", gridTemplateColumns: col2, gap: 32, marginBottom: 24}}>
              <Sl label="Momentum" min={0.6} max={0.98} step={0.001} vk="momentum" dec={3} />
              <Sl label="Weight Decay" min={0} max={0.001} step={0.00001} vk="weight_decay" dec={5} />
            </div>
            <div style={{display: "grid", gridTemplateColumns: col2, gap: 32}}>
              <Sl label="Box Loss Gain" min={0.5} max={20} step={0.5} vk="box" dec={1} />
              <Sl label="Class Loss Gain" min={0.1} max={5} step={0.1} vk="cls" dec={1} />
            </div>
          </div>
        </div>

        <div>
          <div className="hd">Advanced Controls</div>
          <div className="card" style={{marginBottom: 0}}>
            <div style={{display: "grid", gridTemplateColumns: col2, gap: 24, marginBottom: 24}}>
              <div className="field"><label className="lbl">Mosaic Closure</label><input type="number" value={hp.close_mosaic} min={0} max={100} onChange={e => sH("close_mosaic", +e.target.value)} /></div>
              <div className="field"><label className="lbl">Autosave Int.</label><input type="number" value={hp.save_period} min={-1} max={100} onChange={e => sH("save_period", +e.target.value)} /></div>
            </div>
            <div className="g2" style={{gap: 16}}>
              {[["AMP", "amp"], ["Multi-scale", "multi_scale"], ["Auto-Val", "val"], ["Plotting", "plots"]].map(([l, k]) => (
                <label key={k} className="chk">
                  <input type="checkbox" checked={!!hp[k]} onChange={e => sH(k, e.target.checked)} />
                  {l}
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>,

    // STEP 4 — Generate
    <div key="s4">
      {backOk === false && <div className="warn" style={{marginBottom: 32}}>⚠ Backend not connected at {API} — offline mode. Configuration will be downloaded locally.</div>}


      <div style={{display: "grid", gridTemplateColumns: col2, gap: isMobile ? 16 : 32, alignItems: "start"}}>
        <div>
          <div className="hd">Pipeline Blueprint</div>
          <div className="card" style={{marginBottom: 0}}>
            <div className="sgrid">
              {[
                ["Task", task?.label || "—"], ["Architecture", model?.n || "—"], ["Strategy", mode === "scratch" ? "SCRATCH" : "FINE-TUNE"],
                ["Epochs", hp.epochs], ["Batch Size", hp.batch === -1 ? "Dynamic" : hp.batch], ["Resolution", imgSz + "px"],
                ["Optimizer", hp.optimizer || "Auto"], ["Base LR", hp.lr0], ["Dimensions", `${trainSplit}/${valSplit}/${testSplit}`],
                ["Ontology", classes.length || "Auto-detect"], ["Augments", augOnCount + " active"], ["Annotation", FORMAT_GUIDE[fmt]?.label]
              ].map(([k, v]) => (
                <div key={k} className="sitem" style={{background: "var(--bg-tertiary)", border: "none", padding: "16px"}}>
                  <div className="sk" style={{fontSize: 9, marginBottom: 4}}>{k.toUpperCase()}</div>
                  <div className="sv" style={{fontSize: 14}}>{String(v)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div>
          <div className="hd">Deployment Target</div>
          <div style={{display: "flex", flexDirection: "column", gap: 16}}>
            {[
              ["colab", "Google Colab", "Research focus. Mounts Drive.", "Free T4 / A100"],
              ["kaggle", "Kaggle Kernels", "30h training per week.", "Free P100"]
            ].map(([id, n, s, hw]) => (
              <div key={id} className={`platcard${platform === id ? " sel" : ""}`} onClick={() => setPlat(id)} style={{textAlign: "left", padding: 24}}>
                <div style={{display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12}}>
                  <div className="plat-n" style={{fontSize: 16}}>{n}</div>
                  <div className="tag" style={{fontSize: 9}}>{hw}</div>
                </div>
                <div className="plat-s" style={{fontSize: 13, lineHeight: 1.4, color: "var(--text-tertiary)"}}>{s}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{marginTop: 48, textAlign: "center"}}>
         <div style={{fontSize: 13, color: "var(--text-tertiary)", maxWidth: 400, margin: "0 auto"}}>
           Finalize your architecture and use the <strong>Generate Configuration</strong> button in the navigation bar to output your training environment.
         </div>
      </div>

      {logs.length>0&&(
        <div className="card">
          <div className="hd">Pipeline Log</div>
          <div className="logbox">
            {logs.map((l,i)=>(
              <div key={i} className={l.startsWith("✅")||l.startsWith("🚀")?"lok":l.startsWith("⚠")||l.startsWith("💡")?"lwrn":l.startsWith("❌")?"lerr":"linf"}>{l}</div>
            ))}
          </div>
        </div>
      )}

      {result&&(
        <>
          {result.errors?.length>0&&<div className="warn" style={{color:"#fca5a5",background:"rgba(239,68,68,.07)",borderColor:"rgba(239,68,68,.2)"}}>❌ {result.errors.join("\n")}</div>}
          {result.validation?.warnings?.map((w,i)=><div key={i} className="warn">⚠ {w}</div>)}
          {result.validation?.suggestions?.map((s,i)=><div key={i} className="info">💡 {s}</div>)}

          <div className="card">
            <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:9}}>
              <div className="hd" style={{marginBottom:0}}>Generated Files</div>
              <button className="btn btn-g btn-sm" onClick={()=>navigator.clipboard.writeText(outTab==="notebook"?getNotebook():outTab==="yaml"?result.data_yaml:result.readme_md)}>Copy</button>
            </div>
            <div className="otabs">
              {[["notebook","📓 Notebook"],["yaml","📄 data.yaml"],["readme","📋 README"]].map(([id,l])=>(
                <span key={id} className={`otab${outTab===id?" sel":""}`} onClick={()=>setOutTab(id)}>{l}</span>
              ))}
            </div>
            <div className="code">
              {outTab==="notebook"&&(getNotebook().slice(0,3200)+(getNotebook().length>3200?"\n\n… download for full file":""))}
              {outTab==="yaml"&&result.data_yaml}
              {outTab==="readme"&&result.readme_md}
            </div>
          </div>

          <div className="card" style={{background: "rgba(217, 70, 168, 0.03)", border: "1px solid var(--border-color)"}}>
            <div className="hd">Assets & Artifacts</div>
            {[
              [`📓 ${platform === "colab" ? "Colab" : "Kaggle"} Notebook`, "Training environment script", () => dl(getNotebook(), `yoloforge_${model?.id}_${platform}.ipynb`)],
              ["📄 data.yaml", "Class ontology configuration", () => dl(result.data_yaml || "", "data.yaml", "text/plain")],
              ["📋 README.md", "Integration & export docs", () => dl(result.readme_md || "", "README.md", "text/plain")],
            ].map(([n, d, fn]) => (
              <div key={n} className="dlrow">
                <div><div className="dl-n">{n}</div><div className="dl-d">{d}</div></div>
                <button className="btn btn-g btn-sm" onClick={fn} style={{borderRadius: 8}}>DOWNLOAD</button>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="card">
        <div className="hd">{platform==="colab"?"Colab":"Kaggle"} Instructions</div>
        <ul className="slist">
          {(platform==="colab"?[
            <>Go to <code>colab.research.google.com</code></>,
            <>File → Upload notebook → select the <code>.ipynb</code></>,
            <>Runtime → Change runtime type → <code>GPU</code> (T4 recommended)</>,
            <>Run All (<code>Ctrl+F9</code>) — upload dataset ZIP when prompted</>,
            <>Model auto-saved to Drive under <code>yolo_runs/</code></>,
          ]:[
            <>Go to <code>kaggle.com/code</code> → New Notebook</>,
            <>File → Import Notebook → upload the <code>.ipynb</code></>,
            <>Add Data → attach your dataset</>,
            <>Settings → Accelerator → <code>GPU P100</code></>,
            <>Run All — results in <code>/kaggle/working/</code></>,
          ]).map((s,i)=>(
            <li key={i} className="sli"><div className="sn">{i+1}</div><div className="st2">{s}</div></li>
          ))}
        </ul>
      </div>
    </div>,
  ];

  const [started, setStarted] = useState(false);

  return (
    <div style={{minHeight:"100vh", position: "relative"}}>
      <style>{CSS}</style>

      {/* Background Video Wallpaper */}
      <div style={{position: "fixed", top: 0, left: 0, width: "100%", height: "100%", zIndex: 0, overflow: "hidden"}}>
        <video 
          autoPlay 
          loop 
          muted 
          playsInline 
          style={{width: "100%", height: "100%", objectFit: "cover"}}
        >
          <source src="/hero-bg.mp4" type="video/mp4" />
        </video>
      </div>

      {/* Header */}
      <header className="hdr">
        <div className="hdr-in">
          <a href="/" className="logo">
            <div className="logo-box">Y</div>
            <div className="logo-name">YOLO<em>Forge</em></div>
          </a>
          <div style={{display: "flex", gap: 12, alignItems: "center"}}>
            {backOk === true && <span className="tag" style={{borderColor: "#10B981", color: "#10B981", border: "none", fontSize: 11, fontWeight: 700}}>● ONLINE</span>}
            {backOk === false && <span className="tag" style={{borderColor: "#F59E0B", color: "#F59E0B", border: "none", fontSize: 11, fontWeight: 700}}>○ OFFLINE</span>}
          </div>
        </div>
      </header>

      {/* Landing Page (Alone) */}
      {!started ? (
        <section className="hero" style={{height: "100vh", display: "flex", flexDirection: "column", justifyContent: "center", marginBottom: 0}}>

          <h1>Design your <span>Computer Vision</span> pipeline</h1>
          <p>The most advanced agentic platform to generate production-ready YOLO training environments in seconds.</p>
          
          <div style={{marginTop: 48, padding: "0 16px"}}>
            <button className="btn btn-p" onClick={() => setStarted(true)} style={{padding: isMobile ? "16px 32px" : "20px 60px", fontSize: isMobile ? 16 : 20, borderRadius: 100, boxShadow: "0 20px 40px rgba(217, 70, 168, 0.25)", width: isMobile ? "100%" : "auto"}}>
              Get Started — Forging v2.0
            </button>
            <div style={{marginTop: 24, display: "flex", gap: isMobile ? 16 : 32, justifyContent: "center", color: "rgba(255,255,255,0.7)", fontSize: isMobile ? 12 : 14, fontWeight: 600, flexWrap: "wrap"}}>
               <span>✓ 5-Step Pipeline</span>
               <span>✓ 10+ Architectures</span>
               <span>✓ Live Simulator</span>
            </div>
          </div>
        </section>
      ) : (
        <>
          {/* Progress Header */}
          <div style={{textAlign:"center", paddingTop: isMobile ? 72 : 120, paddingBottom: isMobile ? 20 : 60, position: "relative", zIndex: 1}}>
             <div className="sbar" style={{margin: "0 auto", overflowX: "auto"}}>
                {STEPS.map((s, i) => (
                  <div key={s} className={`sit${step === i ? " act" : ""}${step > i ? " done" : ""}`} onClick={() => setStep(i)}>
                    {!isMobile && <span className="sit-n">0{i + 1}</span>} {isMobile ? s.split(" ")[0] : s}
                  </div>
                ))}
              </div>
          </div>

          {/* Main Content with Bento Grid */}
          <main className="wrap" style={{paddingTop: isMobile ? 0 : 80, position: "relative", zIndex: 1}}>
            <div className="bento-grid" style={{
              display: "grid",
              gridTemplateColumns: isMobile ? "1fr" : "repeat(12, 1fr)",
              gap: isMobile ? 14 : 24,
            }}>
              {/* Main Config Card */}
              <div className="bento-card" style={{gridColumn: isMobile ? "span 1" : "span 8"}}>
                {stepContent[step]}
              </div>

              {/* Sidebar Area */}
              <div style={{
                gridColumn: isMobile ? "span 1" : "span 4",
                display: "grid",
                gridTemplateColumns: isMobile ? "1fr 1fr 1fr" : "1fr",
                gap: isMobile ? 10 : 16,
              }}>
                <div className="bento-card" style={{padding: isMobile ? 14 : 20}}>
                  <div className="hd" style={{marginBottom: 12, fontSize: 10}}>Model Context</div>
                  <div style={{display:"flex",flexDirection:"column",gap:isMobile?8:12}}>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Task</span>
                      <span style={{fontWeight:700,color:"var(--text-primary)",textAlign:"right",wordBreak:"break-word",fontSize:isMobile?11:13}}>{task?.short || task?.label || "-"}</span>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Model</span>
                      <span style={{fontWeight:700,color:"var(--text-primary)",textAlign:"right",wordBreak:"break-word",fontSize:isMobile?11:13}}>{model?.n || "-"}</span>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Mode</span>
                      <span className="tag" style={{background:"rgba(217,70,168,0.1)",color:"var(--accent-primary)",border:"none",padding:"1px 6px",fontSize:isMobile?9:11}}>{mode}</span>
                    </div>
                  </div>
                </div>

                <div className="bento-card" style={{padding: isMobile ? 14 : 20}}>
                  <div className="hd" style={{marginBottom: 12, fontSize: 10}}>Data & Vision</div>
                  <div style={{display:"flex",flexDirection:"column",gap:isMobile?8:12}}>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Format</span>
                      <span style={{fontWeight:700,color:"var(--text-primary)",textAlign:"right",fontSize:isMobile?10:13}}>{isMobile ? fmt.toUpperCase() : FORMAT_GUIDE[fmt]?.label || "-"}</span>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Img</span>
                      <span style={{fontWeight:700,color:"var(--text-primary)",fontSize:isMobile?11:13}}>{imgSz}px</span>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Augs</span>
                      <span style={{fontWeight:700,color:(augOnCount>0?"var(--accent-primary)":"var(--text-tertiary)"),fontSize:isMobile?11:13}}>{augOnCount}</span>
                    </div>
                  </div>
                </div>

                <div className="bento-card" style={{padding: isMobile ? 14 : 20}}>
                  <div className="hd" style={{marginBottom: 12, fontSize: 10}}>Train Params</div>
                  <div style={{display:"flex",flexDirection:"column",gap:isMobile?8:12}}>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Epochs</span>
                      <span style={{fontWeight:700,color:"var(--text-primary)",fontSize:isMobile?11:13}}>{hp.epochs}</span>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Batch</span>
                      <span style={{fontWeight:700,color:"var(--text-primary)",fontSize:isMobile?11:13}}>{hp.batch}</span>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:isMobile?11:13,gap:4}}>
                      <span style={{color:"var(--text-tertiary)",flexShrink:0}}>Split</span>
                      <span style={{fontWeight:700,color:"var(--text-primary)",fontSize:isMobile?10:13}}>{trainSplit}/{valSplit}/{testSplit}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </main>

          {/* Persistent Navigation bar */}
          <nav className="nb">
            <div className="nb-in">
              <div style={{display: "flex", gap: 8}}>
                <button className="btn btn-g" disabled={step === 0} onClick={() => setStep(s => s - 1)}>
                  ← {!isMobile && "Back"}
                </button>
                {step < 4 ? (
                  <button className="btn btn-g" onClick={() => setStep(s => s + 1)}>
                    {!isMobile && "Next"} →
                  </button>
                ) : (
                  <button className="btn btn-gen" onClick={handleGenerate} disabled={loading || !task || !model} style={{boxShadow: "0 10px 25px rgba(217, 70, 168, 0.2)", padding: isMobile ? "12px 16px" : undefined, fontSize: isMobile ? 13 : undefined}}>
                    {loading ? <><span className="spin"/>Forging...</> : isMobile ? "Generate ⚡" : "Generate Configuration ⚡"}
                  </button>
                )}
              </div>
              <div style={{display: "flex", gap: isMobile ? 8 : 20, alignItems: "center"}}>
                 <span style={{fontSize: isMobile ? 10 : 12, fontWeight: 700, color: "var(--text-tertiary)"}}>{isMobile ? `${step+1}/5` : `PHASE ${step + 1} OF 5`}</span>
              </div>
            </div>
          </nav>
        </>
      )}
    </div>
  );
}
