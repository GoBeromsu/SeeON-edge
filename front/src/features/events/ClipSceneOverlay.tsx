import type { ClipScene, ClipSceneFrame } from '@/shared/api/types';

type Props = { scene: ClipScene; frame: ClipSceneFrame | null };

export function ClipSceneOverlay({ scene, frame }: Props): JSX.Element | null {
  if (frame === null) return null;
  const [width, height] = scene.source_dimensions;
  const palette = scene.style.palette;
  const layers = [
    ...frame.bd.map((bed) => ({
      z: scene.style.z_order.bed,
      key: `bed-${bed.i}`,
      node: <polygon points={bed.pg.map(([x, y]) => `${x},${y}`).join(' ')} fill="none" stroke={toColor(palette.bed)} strokeWidth="2" />,
    })),
    ...frame.ps.map((person) => ({
      z: scene.style.z_order.person,
      key: `person-${person.i}`,
      node: <Person person={person} scene={scene} />,
    })),
    ...frame.lb.map((label, index) => ({
      z: label.z,
      key: `label-${label.z}-${index}-${label.t}`,
      node: <text x={label.x} y={label.y} fill={toColor(label.c)} fontSize="14">{label.t}</text>,
    })),
  ].sort((left, right) => left.z - right.z);
  return <svg
    viewBox={`0 0 ${width} ${height}`}
    preserveAspectRatio="none"
    className="pointer-events-none absolute inset-0 h-full w-full"
    aria-hidden="true"
    data-testid="clip-scene-overlay"
  >
    {layers.map((layer) => <g key={layer.key}>{layer.node}</g>)}
  </svg>;
}

function Person({ person, scene }: { person: ClipSceneFrame['ps'][number]; scene: ClipScene }): JSX.Element {
  const [x, y, width, height] = person.b;
  const keypoints = new Map(person.k?.filter((point) => point[1] !== null && point[2] !== null).map((point) => [point[0], point]) ?? []);
  return <>
    <rect x={x} y={y} width={width} height={height} fill="none" stroke={toColor(scene.style.palette.person)} strokeWidth="2" />
    {scene.style.skeleton.edges.map(([from, to]) => {
      const start = keypoints.get(from);
      const end = keypoints.get(to);
      return start && end ? <line key={`${from}-${to}`} x1={start[1]!} y1={start[2]!} x2={end[1]!} y2={end[2]!} stroke={toColor(scene.style.palette.pose)} strokeWidth="1.5" opacity={Math.min(start[3], end[3])} /> : null;
    })}
    {[...keypoints.values()].map(([index, pointX, pointY, confidence]) => <circle key={index} cx={pointX!} cy={pointY!} r="2" fill={toColor(scene.style.palette.pose_dot)} opacity={confidence} />)}
  </>;
}

function toColor([red, green, blue]: [number, number, number]): string {
  return `rgb(${red} ${green} ${blue})`;
}
