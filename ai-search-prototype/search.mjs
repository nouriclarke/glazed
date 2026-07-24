import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import readline from 'node:readline';
import mysql from 'mysql2/promise';
import {
  AutoProcessor,
  AutoTokenizer,
  CLIPModel,
  RawImage,
  env,
} from '@xenova/transformers';

const OUT_DIR = path.resolve('sample-images');
const CACHE_DIR = path.resolve('model-cache');
const EMBEDDING_DIR = path.resolve('embedding-cache');
const VISUAL_CACHE_DIR = path.resolve('visual-cache');
const TRAINING_DIR = path.resolve('training-data');
const RANKING_WEIGHTS_PATH = path.join(TRAINING_DIR, 'ranking-weights.json');
const MODEL_NAME = process.env.CLIP_MODEL || 'Xenova/clip-vit-base-patch16';
const LIMIT_VALUE = numberEnv('LIMIT', 0);
const LIMIT = LIMIT_VALUE > 0 ? Math.floor(LIMIT_VALUE) : null;
const TOP_K = Math.max(1, Math.floor(numberEnv('TOP_K', 10)));
const EMBEDDING_PREFILTER = Math.max(0, Math.floor(numberEnv('EMBEDDING_PREFILTER', 60)));
const ARGS = process.argv.slice(2);
const STDIO_MODE = ARGS.includes('--stdio');
const QUERY = ARGS.filter((arg) => arg !== '--stdio').join(' ').trim();
let workerRowsCache = null;
let workerRowsRevision = null;

env.cacheDir = CACHE_DIR;
env.localModelPath = CACHE_DIR;
env.allowLocalModels = true;
env.allowRemoteModels = process.env.ALLOW_REMOTE === '1';

function numberEnv(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) ? value : fallback;
}

function fail(message, details = undefined) {
  console.error(details ? `${message}\n${details}` : message);
  process.exit(1);
}

if (!STDIO_MODE && !QUERY) {
  fail('Please provide a search prompt.');
}

function hostDatabaseValue(value = process.env.DB_HOST || '127.0.0.1') {
  return value === 'tile-db' ? '127.0.0.1' : value;
}

function labToRgb(l, a, b) {
  const y = (l + 16) / 116;
  const x = a / 500 + y;
  const z = y - b / 200;
  const xyz = [x, y, z].map((v) => {
    const v3 = v ** 3;
    return v3 > 0.008856 ? v3 : (v - 16 / 116) / 7.787;
  });

  let [X, Y, Z] = [xyz[0] * 95.047, xyz[1] * 100, xyz[2] * 108.883];
  X /= 100;
  Y /= 100;
  Z /= 100;

  let r = X * 3.2406 + Y * -1.5372 + Z * -0.4986;
  let g = X * -0.9689 + Y * 1.8758 + Z * 0.0415;
  let bl = X * 0.0557 + Y * -0.204 + Z * 1.057;

  const correct = (v) => (v > 0.0031308 ? 1.055 * v ** (1 / 2.4) - 0.055 : 12.92 * v);
  return [r, g, bl].map((v) => Math.max(0, Math.min(255, Math.round(correct(v) * 255))));
}

function rgbToLuminance(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function rgbToHsv(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;

  if (delta !== 0) {
    if (max === r) h = ((g - b) / delta) % 6;
    else if (max === g) h = (b - r) / delta + 2;
    else h = (r - g) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
  }

  return [h, max === 0 ? 0 : delta / max, max];
}

function rgbFamily(r, g, b) {
  const [hue, saturation, value] = rgbToHsv(r, g, b);
  if (value < 0.16) return 'black';
  if (saturation < 0.09 && value > 0.84) return 'white';
  if (saturation < 0.15) {
    if (r > b + 8 && g > b + 2) return 'cream';
    return 'gray';
  }
  if (hue >= 18 && hue <= 55 && value < 0.72) return 'brown';
  if (hue >= 35 && hue <= 68 && value > 0.72 && saturation < 0.45) return 'cream';
  if (hue < 15 || hue >= 345) return 'red';
  if (hue < 35) return 'orange';
  if (hue < 68) return 'yellow';
  if (hue < 165) return 'green';
  if (hue < 195) return 'cyan';
  if (hue < 255) return 'blue';
  if (hue < 292) return 'purple';
  if (hue < 345) return 'pink';
  return 'unknown';
}

function hueDistance(a, b) {
  const difference = Math.abs(a - b) % 360;
  return Math.min(difference, 360 - difference);
}

function rgbColorFamilyScore(family, r, g, b) {
  const [hue, saturation, value] = rgbToHsv(r, g, b);
  const saturationScore = clamp01((saturation - 0.04) / 0.36);
  const hueScore = (target, tolerance) => clamp01(1 - hueDistance(hue, target) / tolerance);

  switch (family) {
    case 'black':
      return clamp01((0.3 - value) / 0.24);
    case 'white':
      return clamp01((value - 0.72) / 0.24) * clamp01((0.18 - saturation) / 0.15);
    case 'gray':
      return clamp01((0.2 - saturation) / 0.16) * clamp01(1 - Math.abs(value - 0.55) / 0.5);
    case 'cream':
      return hueScore(45, 48)
        * clamp01((value - 0.62) / 0.28)
        * clamp01(1 - Math.abs(saturation - 0.16) / 0.2);
    case 'brown': {
      const brownHue = hueScore(35, 38);
      const brownSaturation = clamp01((saturation - 0.07) / 0.28);
      const brightTanAllowance = 1 - clamp01((value - 0.86) / 0.14) * 0.55;
      return brownHue * brownSaturation * brightTanAllowance;
    }
    case 'red':
      return hueScore(0, 38) * saturationScore;
    case 'orange':
      return hueScore(28, 28) * saturationScore;
    case 'yellow':
      return hueScore(55, 28) * saturationScore;
    case 'green':
      return hueScore(120, 62) * saturationScore;
    case 'cyan':
      return hueScore(180, 42) * saturationScore;
    case 'blue':
      return hueScore(225, 55) * saturationScore;
    case 'purple':
      return hueScore(275, 42) * saturationScore;
    case 'pink':
      return hueScore(325, 38) * saturationScore;
    default:
      return 0;
  }
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function targetLightnessScore(lightness, target, tolerance = 0.42) {
  return clamp01(1 - Math.abs(clamp01(lightness) - target) / tolerance);
}

function normalize(vec) {
  const norm = Math.sqrt(vec.reduce((sum, value) => sum + value * value, 0)) || 1;
  return Array.from(vec, (value) => value / norm);
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

function legacyColorIntentScore(query, row) {
  if (!hasLegacyLab(row)) return 0;

  const tokens = queryTokens(query);
  const [r, g, b] = labToRgb(Number(row.Color_L), Number(row.Color_A), Number(row.Color_B));
  const requestedFamilies = new Set(tokens.map((token) => FAMILY_ALIASES[token]).filter(Boolean));
  const warmth = Math.max(0, (r + 0.45 * g - 1.2 * b) / 255);
  const coolness = Math.max(0, (b + 0.35 * g - 1.05 * r) / 255);
  const brightness = Number(row.Color_L) / 100;
  let score = 0;
  let factors = 0;

  for (const family of requestedFamilies) {
    score += rgbColorFamilyScore(family, r, g, b);
    factors += 1;
  }
  if (tokens.includes('warm')) {
    score += warmth;
    factors += 1;
  }
  if (tokens.includes('cold') || tokens.includes('cool')) {
    score += coolness;
    factors += 1;
  }
  if (tokens.some((token) => ['bright', 'light', 'pale'].includes(token))) {
    score += brightness;
    factors += 1;
  }
  if (tokens.some((token) => ['lighter', 'brighter'].includes(token))) {
    score += targetLightnessScore(brightness, 0.68);
    factors += 1;
  }
  if (tokens.includes('dark') || tokens.includes('deep')) {
    score += 1 - brightness;
    factors += 1;
  }
  if (tokens.includes('darker') || tokens.includes('deeper')) {
    score += targetLightnessScore(brightness, 0.38);
    factors += 1;
  }
  if (tokens.some((token) => ['earth', 'earthy', 'rustic'].includes(token))) {
    score += rgbColorFamilyScore('brown', r, g, b);
    factors += 1;
  }

  return factors ? score / factors : 0;
}

const FAMILY_ALIASES = {
  aqua: 'cyan',
  beige: 'cream',
  black: 'black',
  blue: 'blue',
  brown: 'brown',
  burgundy: 'red',
  cream: 'cream',
  cyan: 'cyan',
  gold: 'yellow',
  gray: 'gray',
  green: 'green',
  grey: 'gray',
  maroon: 'red',
  navy: 'blue',
  orange: 'orange',
  pink: 'pink',
  purple: 'purple',
  red: 'red',
  tan: 'brown',
  teal: 'cyan',
  turquoise: 'cyan',
  violet: 'purple',
  white: 'white',
  yellow: 'yellow',
};

function parseJsonField(value, fallback) {
  if (!value) return fallback;
  if (typeof value === 'object') return value;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function normalizeProfile(profile) {
  if (!profile || Array.isArray(profile) || typeof profile !== 'object') return {};

  return Object.fromEntries(Object.entries(profile)
    .map(([family, percent]) => [family.toLowerCase(), Number(percent)])
    .filter(([, percent]) => Number.isFinite(percent) && percent > 0));
}

function dominantColors(row) {
  const parsed = parseJsonField(row.DominantColors, []);
  return Array.isArray(parsed) ? parsed : [];
}

function hasLegacyLab(row) {
  return [row.Color_L, row.Color_A, row.Color_B].every((value) => Number.isFinite(Number(value)));
}

function displayRgb(row) {
  const [firstColor] = dominantColors(row);
  if (Array.isArray(firstColor?.rgb) && firstColor.rgb.length === 3) {
    return firstColor.rgb.map((value) => Math.max(0, Math.min(255, Math.round(Number(value) || 0))));
  }

  if (hasLegacyLab(row)) {
    return labToRgb(Number(row.Color_L), Number(row.Color_A), Number(row.Color_B));
  }

  return [0, 0, 0];
}

function colorProfile(row) {
  return normalizeProfile(parseJsonField(row.ColorProfile, {}));
}

function familyPercent(row, family, profile = colorProfile(row), colors = dominantColors(row)) {
  if (Number.isFinite(profile[family])) {
    return profile[family];
  }

  return Math.max(
    0,
    ...colors
      .filter((color) => color?.family === family)
      .map((color) => Number(color.percent) || 0),
  );
}

function profileGroupPercent(row, families, profile, colors) {
  return families.reduce((sum, family) => sum + familyPercent(row, family, profile, colors), 0);
}

function weightedLightness(row, colors = dominantColors(row)) {
  let weightedTotal = 0;
  let totalWeight = 0;

  for (const color of colors) {
    const lightness = Number(Array.isArray(color.lab) ? color.lab[0] : color.lab?.l);
    const percent = Number(color.percent);
    if (!Number.isFinite(lightness) || !Number.isFinite(percent)) continue;
    weightedTotal += lightness * percent;
    totalWeight += percent;
  }

  if (totalWeight > 0) {
    return weightedTotal / totalWeight / 100;
  }

  return Number.isFinite(Number(row.Color_L)) ? Number(row.Color_L) / 100 : 0.5;
}

function queryTokens(query) {
  return query
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

function parseQueryIntent(query) {
  const lower = query.toLowerCase();
  const noDarkEdges = /\b(?:no|without|avoid|not)\s+(?:dark|black)\s+(?:edge|edges|border|borders|rim|rims|frame|outline)\b/.test(lower);
  const noBrownEdges = /\b(?:no|without|avoid|not)\s+brown\s+(?:edge|edges|border|borders|rim|rims|frame|outline)\b/.test(lower);
  const noEdges = /\b(?:no|without|avoid)\s+(?:edge|edges|border|borders|rim|rims|frame|outline)\b/.test(lower);
  const wantsDarkEdges = !noDarkEdges && /\b(?:dark|black|brown)\s+(?:edge|edges|border|borders|rim|rims|frame|outline)\b/.test(lower);
  const wantsLightEdges = !noEdges && /\b(?:light|pale|cream|white)\s+(?:edge|edges|border|borders|rim|rims|frame|outline)\b/.test(lower);

  const positiveQuery = lower
    .replace(/\b(?:no|without|avoid|not)\s+(?:dark|black|brown)?\s*(?:edge|edges|border|borders|rim|rims|frame|outline)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  return {
    rawQuery: query,
    positiveQuery,
    tokens: queryTokens(positiveQuery),
    noDarkEdges,
    noBrownEdges,
    noEdges,
    wantsDarkEdges,
    wantsLightEdges,
  };
}

function requestedColorFamilies(query) {
  return [...new Set(queryTokens(query).map((token) => FAMILY_ALIASES[token]).filter(Boolean))];
}

function explicitColorMatchScore(families, row) {
  if (families.length === 0) return 0;

  const profile = colorProfile(row);
  const colors = dominantColors(row);
  const [r, g, b] = displayRgb(row);
  return Math.max(...families.map((family) => {
    const profileScore = familyPercent(row, family, profile, colors);
    const primaryBoost = String(row.PrimaryColor || '').toLowerCase() === family ? 0.2 : 0;
    const rgbScore = rgbColorFamilyScore(family, r, g, b);
    return clamp01(Math.max(profileScore + primaryBoost, rgbScore));
  }));
}

function colorIntentScore(query, row) {
  const profile = colorProfile(row);
  const colors = dominantColors(row);
  if (Object.keys(profile).length === 0 && colors.length === 0) {
    return legacyColorIntentScore(query, row);
  }

  const tokens = queryTokens(query);
  const exactFamilies = new Set(tokens.map((token) => FAMILY_ALIASES[token]).filter(Boolean));
  let score = 0;
  let factors = 0;

  for (const family of exactFamilies) {
    const percent = familyPercent(row, family, profile, colors);
    const primaryBoost = row.PrimaryColor === family ? 0.25 : 0;
    score += Math.min(1.25, percent * 1.3 + primaryBoost);
    factors += 1;
  }

  if (tokens.some((token) => ['warm', 'earth', 'earthy', 'rustic'].includes(token))) {
    score += Math.min(1.2, profileGroupPercent(row, ['red', 'orange', 'yellow', 'brown', 'cream'], profile, colors) * 1.2);
    factors += 1;
  }

  if (tokens.some((token) => ['cold', 'cool'].includes(token))) {
    score += Math.min(1.2, profileGroupPercent(row, ['blue', 'cyan', 'green', 'purple'], profile, colors) * 1.2);
    factors += 1;
  }

  if (tokens.some((token) => ['sea', 'ocean', 'water', 'marine'].includes(token))) {
    score += Math.min(1.2, profileGroupPercent(row, ['blue', 'cyan', 'green'], profile, colors) * 1.25);
    factors += 1;
  }

  const lightness = weightedLightness(row, colors);
  if (tokens.some((token) => ['bright', 'light', 'pale'].includes(token))) {
    score += lightness;
    factors += 1;
  }

  if (tokens.some((token) => ['lighter', 'brighter'].includes(token))) {
    score += targetLightnessScore(lightness, 0.68);
    factors += 1;
  }

  if (tokens.includes('dark') || tokens.includes('deep')) {
    score += 1 - lightness;
    factors += 1;
  }

  if (tokens.includes('darker') || tokens.includes('deeper')) {
    score += targetLightnessScore(lightness, 0.38);
    factors += 1;
  }

  return factors ? score / factors : 0;
}

const STOP_WORDS = new Set([
  'a', 'an', 'and', 'for', 'like', 'looking', 'of', 'that', 'the', 'tile', 'tiles',
  'with', 'feeling', 'feel', 'kind', 'sort', 'show', 'me', 'want',
  'no', 'not', 'without', 'avoid',
]);

const TOKEN_ALIASES = {
  cold: ['cool', 'blue', 'cyan'],
  cool: ['cold', 'blue', 'cyan'],
  warm: ['earthy', 'brown', 'orange', 'red', 'cream'],
  earth: ['earthy', 'brown', 'rustic'],
  earthy: ['earth', 'brown', 'rustic'],
  rustic: ['earthy', 'brown', 'rough'],
  shiny: ['glossy', 'highlighted'],
  reflective: ['glossy', 'highlighted'],
  gloss: ['glossy', 'highlighted'],
  rough: ['speckled', 'variegated', 'high contrast'],
  speckled: ['variegated', 'rough', 'high contrast'],
  variegated: ['speckled', 'rough', 'high contrast'],
  texture: ['speckled', 'variegated', 'rough'],
  textured: ['speckled', 'variegated', 'rough'],
  spotty: ['speckled', 'variegated'],
  spotted: ['speckled', 'variegated'],
  pale: ['light', 'cream', 'white'],
  lighter: ['light', 'cream'],
  brighter: ['light', 'cream', 'glossy'],
  darker: ['muted', 'deeper'],
  deeper: ['muted', 'darker'],
  beige: ['cream', 'brown', 'warm'],
  sea: ['blue', 'cyan', 'cool'],
  ocean: ['blue', 'cyan', 'cool'],
  water: ['blue', 'cyan', 'cool'],
  marine: ['blue', 'cyan', 'cool'],
};

function metadataScore(query, row) {
  const qTokens = new Set(query
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token && !STOP_WORDS.has(token)));
  const text = [
    row.GlazeType,
    row.SurfaceCondition,
    row.FiringType,
    row.SoilType,
    row.ChemicalComposition,
    row.AutoTags,
    row.AutoKeywords,
  ].join(' ').toLowerCase();

  if (!text.trim()) return 0;

  let score = 0;
  for (const token of qTokens) {
    if (text.includes(token)) {
      score += 1;
      continue;
    }

    const aliases = TOKEN_ALIASES[token] || [];
    if (aliases.some((alias) => text.includes(alias))) {
      score += 0.8;
    }
  }

  return Math.min(score / Math.max(qTokens.size, 1), 1.5);
}

async function loadVisualCache() {
  await fs.mkdir(VISUAL_CACHE_DIR, { recursive: true });
  const filePath = path.join(VISUAL_CACHE_DIR, 'visual-metrics.json');
  if (!(await fileExists(filePath))) {
    return { filePath, metrics: {} };
  }

  const parsed = JSON.parse(await fs.readFile(filePath, 'utf8'));
  return { filePath, metrics: parsed.metrics || {} };
}

async function saveVisualCache(filePath, metrics) {
  await fs.writeFile(
    filePath,
    JSON.stringify({ updatedAt: new Date().toISOString(), metrics }),
  );
}

function visualMetricsFromImage(image) {
  const { data, width, height, channels } = image;
  const marginX = Math.max(1, Math.floor(width * 0.14));
  const marginY = Math.max(1, Math.floor(height * 0.14));
  const centerLeft = Math.floor(width * 0.24);
  const centerRight = Math.ceil(width * 0.76);
  const centerTop = Math.floor(height * 0.24);
  const centerBottom = Math.ceil(height * 0.76);
  const step = Math.max(1, Math.floor(Math.sqrt((width * height) / 9000)));

  let borderCount = 0;
  let centerCount = 0;
  let allCount = 0;
  let borderLum = 0;
  let centerLum = 0;
  let allLum = 0;
  let darkBorder = 0;
  let brownBorder = 0;
  let blackBorder = 0;
  let darkAll = 0;
  let centerDark = 0;

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const offset = (y * width + x) * channels;
      const r = data[offset];
      const g = data[offset + 1];
      const b = data[offset + 2];
      const lum = rgbToLuminance(r, g, b);
      const family = rgbFamily(r, g, b);
      const isBorder = x < marginX || x >= width - marginX || y < marginY || y >= height - marginY;
      const isCenter = x >= centerLeft && x < centerRight && y >= centerTop && y < centerBottom;

      allCount += 1;
      allLum += lum;
      if (lum < 80) darkAll += 1;

      if (isBorder) {
        borderCount += 1;
        borderLum += lum;
        if (lum < 95) darkBorder += 1;
        if (family === 'brown') brownBorder += 1;
        if (family === 'black') blackBorder += 1;
      }

      if (isCenter) {
        centerCount += 1;
        centerLum += lum;
        if (lum < 80) centerDark += 1;
      }
    }
  }

  const borderLightness = borderCount ? borderLum / borderCount / 255 : 0.5;
  const centerLightness = centerCount ? centerLum / centerCount / 255 : 0.5;
  const darkBorderRatio = borderCount ? darkBorder / borderCount : 0;
  const brownBorderRatio = borderCount ? brownBorder / borderCount : 0;
  const blackBorderRatio = borderCount ? blackBorder / borderCount : 0;
  const darkPixelRatio = allCount ? darkAll / allCount : 0;
  const centerDarkRatio = centerCount ? centerDark / centerCount : 0;
  const borderContrast = Math.max(0, centerLightness - borderLightness);
  const edgeFrameScore = clamp01(darkBorderRatio * 0.55 + borderContrast * 0.9 + brownBorderRatio * 0.25 + blackBorderRatio * 0.35);

  return {
    borderLightness: Number(borderLightness.toFixed(4)),
    centerLightness: Number(centerLightness.toFixed(4)),
    borderContrast: Number(borderContrast.toFixed(4)),
    darkBorderRatio: Number(darkBorderRatio.toFixed(4)),
    brownBorderRatio: Number(brownBorderRatio.toFixed(4)),
    blackBorderRatio: Number(blackBorderRatio.toFixed(4)),
    darkPixelRatio: Number(darkPixelRatio.toFixed(4)),
    centerDarkRatio: Number(centerDarkRatio.toFixed(4)),
    edgeFrameScore: Number(edgeFrameScore.toFixed(4)),
  };
}

async function enrichVisualMetrics(rows) {
  const { filePath, metrics } = await loadVisualCache();
  let changed = false;

  for (const row of rows) {
    const key = cacheKeyFor(row);
    if (!metrics[key]) {
      const image = await RawImage.read(row.imagePath);
      metrics[key] = visualMetricsFromImage(image);
      changed = true;
    }
    row.visualMetrics = metrics[key];
  }

  const activeKeys = new Set(rows.map(cacheKeyFor));
  for (const key of Object.keys(metrics)) {
    if (!activeKeys.has(key)) {
      delete metrics[key];
      changed = true;
    }
  }

  if (changed) {
    await saveVisualCache(filePath, metrics);
  }
}

function visualIntentScore(intent, row) {
  const metrics = row.visualMetrics || {};
  const edgeFrameScore = Number(metrics.edgeFrameScore) || 0;
  const darkBorderRatio = Number(metrics.darkBorderRatio) || 0;
  const brownBorderRatio = Number(metrics.brownBorderRatio) || 0;
  const blackBorderRatio = Number(metrics.blackBorderRatio) || 0;
  let score = 0;
  let factors = 0;
  let penalty = 0;

  if (intent.wantsDarkEdges) {
    score += edgeFrameScore;
    factors += 1;
  }

  if (intent.wantsLightEdges) {
    score += 1 - edgeFrameScore;
    factors += 1;
  }

  if (intent.noDarkEdges) {
    penalty += edgeFrameScore * 1.05 + darkBorderRatio * 0.65 + blackBorderRatio * 0.45 + brownBorderRatio * 0.28;
  }

  if (intent.noBrownEdges) {
    penalty += brownBorderRatio * 0.8 + edgeFrameScore * 0.25;
  }

  if (intent.noEdges) {
    penalty += edgeFrameScore * 0.85 + darkBorderRatio * 0.35 + brownBorderRatio * 0.3 + blackBorderRatio * 0.35;
  }

  return {
    score: factors ? score / factors : 0,
    penalty: clamp01(penalty),
  };
}

function heuristicRankScore(intent, row) {
  const colorScore = colorIntentScore(intent.positiveQuery, row);
  const textScore = metadataScore(intent.positiveQuery, row);
  const visualScore = visualIntentScore(intent, row);
  return colorScore * 0.42 + textScore * 0.22 + visualScore.score * 0.24 - visualScore.penalty * 0.9;
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function loadEmbeddingCache() {
  await fs.mkdir(EMBEDDING_DIR, { recursive: true });
  const filePath = path.join(EMBEDDING_DIR, `${MODEL_NAME.replaceAll('/', '__')}.json`);
  if (!(await fileExists(filePath))) {
    return { filePath, embeddings: {} };
  }

  const parsed = JSON.parse(await fs.readFile(filePath, 'utf8'));
  return { filePath, embeddings: parsed.embeddings || {} };
}

let modelStatePromise = null;
let embeddingCacheStatePromise = null;
let rankingWeightsPromise = null;
let textImageInputsCacheKey = '';
let textImageInputsPromise = null;
const textEmbeddingCache = new Map();
const TEXT_EMBEDDING_CACHE_LIMIT = 48;
const DEFAULT_RANKING_WEIGHTS = {
  bias: 0,
  clipScore: 1,
  colorScore: 0.22,
  explicitColorScore: 0.85,
  colorMismatchPenalty: -1.1,
  metadataScore: 0.12,
  visualScore: 0.18,
  visualPenalty: -0.82,
  exclusionPenalty: -1,
};

async function getModelState() {
  if (!modelStatePromise) {
    modelStatePromise = (async () => {
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);
      const processor = await AutoProcessor.from_pretrained(MODEL_NAME);
      const model = await CLIPModel.from_pretrained(MODEL_NAME);
      const dummyText = tokenizer(['a ceramic tile'], { padding: true, truncation: true });
      return { tokenizer, processor, model, dummyText };
    })();
  }

  return modelStatePromise;
}

async function getEmbeddingCacheState() {
  if (!embeddingCacheStatePromise) {
    embeddingCacheStatePromise = loadEmbeddingCache();
  }

  return embeddingCacheStatePromise;
}

async function getRankingWeights() {
  if (!rankingWeightsPromise) {
    rankingWeightsPromise = (async () => {
      if (process.env.IGNORE_TRAINED_WEIGHTS === '1' || !(await fileExists(RANKING_WEIGHTS_PATH))) {
        return { ...DEFAULT_RANKING_WEIGHTS };
      }

      const parsed = JSON.parse(await fs.readFile(RANKING_WEIGHTS_PATH, 'utf8'));
      return {
        ...DEFAULT_RANKING_WEIGHTS,
        ...(parsed.weights || parsed),
      };
    })();
  }

  return rankingWeightsPromise;
}

async function getTextImageInputs(processor, row) {
  const key = cacheKeyFor(row);
  if (textImageInputsCacheKey !== key) {
    textImageInputsCacheKey = key;
    textImageInputsPromise = RawImage
      .read(row.imagePath)
      .then(async (image) => processor(await image.resize(224, 224)));
  }

  return textImageInputsPromise;
}

function rememberTextEmbedding(key, embedding) {
  if (textEmbeddingCache.has(key)) {
    textEmbeddingCache.delete(key);
  }

  textEmbeddingCache.set(key, embedding);

  while (textEmbeddingCache.size > TEXT_EMBEDDING_CACHE_LIMIT) {
    const [oldestKey] = textEmbeddingCache.keys();
    textEmbeddingCache.delete(oldestKey);
  }
}

async function getTextEmbedding(query, rows, modelState) {
  const key = `${MODEL_NAME}:${query}`;
  const cached = textEmbeddingCache.get(key);
  if (cached) {
    textEmbeddingCache.delete(key);
    textEmbeddingCache.set(key, cached);
    return cached;
  }

  const queryText = modelState.tokenizer([query], { padding: true, truncation: true });
  const queryImage = await getTextImageInputs(modelState.processor, rows[0]);
  const queryOutput = await modelState.model({ ...queryText, ...queryImage });
  const embedding = normalize(Array.from(queryOutput.text_embeds.data));
  rememberTextEmbedding(key, embedding);
  return embedding;
}

async function saveEmbeddingCache(filePath, embeddings) {
  await fs.writeFile(
    filePath,
    JSON.stringify({ model: MODEL_NAME, updatedAt: new Date().toISOString(), embeddings }),
  );
}

function cacheKeyFor(row) {
  return `${row.ID}:${row.imageHash}`;
}

async function removeStaleSampleImages(activeImagePaths) {
  const activeFileNames = new Set(activeImagePaths.map((imagePath) => path.basename(imagePath)));
  const files = await fs.readdir(OUT_DIR).catch(() => []);

  await Promise.all(files
    .filter((file) => file.toLowerCase().endsWith('.jpg') && !activeFileNames.has(file))
    .map((file) => fs.rm(path.join(OUT_DIR, file), { force: true })));
}

async function ensureColorMetadataColumns(conn) {
  const requiredColumns = {
    PrimaryColor: 'VARCHAR(32) DEFAULT NULL',
    DominantColors: 'MEDIUMTEXT DEFAULT NULL',
    ColorProfile: 'MEDIUMTEXT DEFAULT NULL',
  };
  const [columns] = await conn.execute(
    `SELECT COLUMN_NAME
     FROM information_schema.COLUMNS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'testpiece'
       AND COLUMN_NAME IN (?, ?, ?)`,
    Object.keys(requiredColumns),
  );
  const existing = new Set(columns.map((column) => column.COLUMN_NAME));

  for (const [column, definition] of Object.entries(requiredColumns)) {
    if (!existing.has(column)) {
      await conn.execute(`ALTER TABLE testpiece ADD COLUMN ${column} ${definition}`);
    }
  }
}

async function loadTiles() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const conn = await mysql.createConnection({
    host: hostDatabaseValue(),
    port: Number(process.env.DB_PORT || 3306),
    user: process.env.DB_USER || 'ceramadmin',
    password: process.env.DB_PASSWORD || 'glazed-dev-password',
    database: process.env.DB_NAME || 'tilearchive',
  });

  await ensureColorMetadataColumns(conn);

  const query = `SELECT
      tp.ID,
      tp.Image,
      tp.Color_L,
      tp.Color_A,
      tp.Color_B,
      tp.PrimaryColor,
      tp.DominantColors,
      tp.ColorProfile,
      gt.Name AS GlazeType,
      sc.Name AS SurfaceCondition,
      tp.FiringType,
      tp.SoilType,
      tp.ChemicalComposition,
      tp.AutoTags,
      tp.AutoKeywords
    FROM testpiece tp
    INNER JOIN (
      SELECT MIN(ID) AS ID
      FROM testpiece
      WHERE Image IS NOT NULL
      GROUP BY SHA2(Image, 256)
    ) canonical ON canonical.ID = tp.ID
    LEFT JOIN glazetype gt ON tp.GlazeTypeID = gt.ID
    LEFT JOIN surfacecondition sc ON tp.SurfaceConditionID = sc.ID
    WHERE tp.Image IS NOT NULL
      AND (
        (tp.Color_L IS NOT NULL AND tp.Color_A IS NOT NULL AND tp.Color_B IS NOT NULL)
        OR tp.ColorProfile IS NOT NULL
        OR tp.DominantColors IS NOT NULL
      )
    ORDER BY tp.ID DESC
    ${LIMIT && LIMIT > 0 ? 'LIMIT ?' : ''}`;

  const [rows] = await conn.execute(
    query,
    LIMIT && LIMIT > 0 ? [LIMIT] : [],
  );
  await conn.end();

  for (const row of rows) {
    const imageBuffer = Buffer.from(row.Image);
    row.imageHash = crypto.createHash('sha256').update(imageBuffer).digest('hex');
    row.imagePath = path.join(OUT_DIR, `${row.ID}-${row.imageHash.slice(0, 16)}.jpg`);
    if (!(await fileExists(row.imagePath))) {
      await fs.writeFile(row.imagePath, imageBuffer);
    }
    delete row.Image;
  }

  await removeStaleSampleImages(rows.map((row) => row.imagePath));

  return rows;
}

async function getTileRevision() {
  const conn = await mysql.createConnection({
    host: hostDatabaseValue(),
    port: Number(process.env.DB_PORT || 3306),
    user: process.env.DB_USER || 'ceramadmin',
    password: process.env.DB_PASSWORD || 'glazed-dev-password',
    database: process.env.DB_NAME || 'tilearchive',
  });
  const [rows] = await conn.execute(
    'SELECT COUNT(*) AS rowCount, COALESCE(MAX(ID), 0) AS maxId FROM testpiece WHERE Image IS NOT NULL',
  );
  await conn.end();
  return `${rows[0].rowCount}:${rows[0].maxId}`;
}

async function loadSearchRows() {
  if (!STDIO_MODE) {
    return loadTiles();
  }

  const revision = await getTileRevision();
  if (workerRowsCache && workerRowsRevision === revision) {
    return workerRowsCache;
  }

  workerRowsCache = await loadTiles();
  workerRowsRevision = revision;
  return workerRowsCache;
}

async function runSearch(query) {
  const rows = await loadSearchRows();
  if (rows.length === 0) {
    return { query, results: [], message: 'No tile images found in MariaDB.' };
  }

  const intent = parseQueryIntent(query);
  await enrichVisualMetrics(rows);

  const modelState = await getModelState();
  const { filePath: embeddingCachePath, embeddings } = await getEmbeddingCacheState();

  let cacheChanged = false;
  const activeCacheKeys = new Set(rows.map(cacheKeyFor));
  for (const key of Object.keys(embeddings)) {
    if (!activeCacheKeys.has(key)) {
      delete embeddings[key];
      cacheChanged = true;
    }
  }

  const missingRows = rows.filter((row) => !embeddings[cacheKeyFor(row)]);
  let rowsToEmbed = missingRows;
  if (EMBEDDING_PREFILTER > 0 && missingRows.length > EMBEDDING_PREFILTER) {
    const candidateKeys = new Set(rows
      .map((row) => ({ row, score: heuristicRankScore(intent, row) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(TOP_K * 6, EMBEDDING_PREFILTER))
      .map(({ row }) => cacheKeyFor(row)));
    rowsToEmbed = missingRows.filter((row) => candidateKeys.has(cacheKeyFor(row)));
  }

  for (const row of rowsToEmbed) {
    const cacheKey = cacheKeyFor(row);
    if (embeddings[cacheKey]) continue;

    const image = await RawImage.read(row.imagePath);
    const imageInputs = await modelState.processor(await image.resize(224, 224));
    const output = await modelState.model({ ...modelState.dummyText, ...imageInputs });
    embeddings[cacheKey] = normalize(Array.from(output.image_embeds.data));
    cacheChanged = true;
  }

  if (cacheChanged) {
    await saveEmbeddingCache(embeddingCachePath, embeddings);
  }

  const textEmbedding = await getTextEmbedding(query, rows, modelState);
  const rankingWeights = await getRankingWeights();
  const globalColorQuery = intent.positiveQuery
    .replace(/\b(?:black|blue|brown|cream|gray|green|grey|orange|pink|purple|red|white|yellow)\s+(?:edge|edges|border|borders|rim|rims|frame|outline)\b/g, ' ')
    .trim();
  const explicitColorFamilies = requestedColorFamilies(globalColorQuery);

  const rankedResults = rows.map((row) => {
    const cachedEmbedding = embeddings[cacheKeyFor(row)];
    const clipScore = cachedEmbedding ? cosine(textEmbedding, cachedEmbedding) : 0;
    const colorScore = colorIntentScore(intent.positiveQuery, row);
    const textScore = metadataScore(intent.positiveQuery, row);
    const visualScore = visualIntentScore(intent, row);
    const lightness = weightedLightness(row);
    const exclusionPenalty = visualScore.penalty > 0.22 ? visualScore.penalty * 0.75 : 0;
    const explicitColorScore = explicitColorMatchScore(explicitColorFamilies, row);
    const colorMismatchPenalty = explicitColorFamilies.length > 0 ? 1 - explicitColorScore : 0;
    const features = {
      bias: 1,
      clipScore,
      colorScore,
      explicitColorScore,
      colorMismatchPenalty,
      metadataScore: textScore,
      visualScore: visualScore.score,
      visualPenalty: visualScore.penalty,
      exclusionPenalty,
    };
    const finalScore = Object.entries(rankingWeights)
      .reduce((sum, [feature, weight]) => sum + (features[feature] || 0) * weight, 0);
    const [r, g, b] = displayRgb(row);

    return {
      id: String(row.ID),
      imageHash: row.imageHash,
      finalScore,
      clipScore,
      colorScore,
      explicitColorScore,
      colorMismatchPenalty,
      metadataScore: textScore,
      visualScore: visualScore.score,
      visualPenalty: visualScore.penalty,
      primaryColor: row.PrimaryColor || '',
      colorProfile: colorProfile(row),
      lightness,
      tags: row.AutoTags || '',
      visualMetrics: row.visualMetrics || {},
      features,
      rgb: [r, g, b],
    };
  })
    .sort((a, b) => b.finalScore - a.finalScore);

  const seenImageHashes = new Set();
  const results = [];
  for (const result of rankedResults) {
    if (seenImageHashes.has(result.imageHash)) {
      continue;
    }

    seenImageHashes.add(result.imageHash);
    results.push(result);
    if (results.length >= TOP_K) {
      break;
    }
  }

  const maxScore = results[0]?.finalScore ?? 0;
  const minScore = results.at(-1)?.finalScore ?? maxScore;
  for (const result of results) {
    const relativeMatchScore = maxScore === minScore
      ? 1
      : 0.55 + ((result.finalScore - minScore) / (maxScore - minScore)) * 0.44;
    const colorConfidence = explicitColorFamilies.length > 0
      ? 0.3 + result.explicitColorScore * 0.7
      : 1;
    result.matchScore = relativeMatchScore * colorConfidence;
  }

  return {
    query,
    resolvedQuery: intent.positiveQuery,
    model: MODEL_NAME,
    searchedRows: rows.length,
    embeddedRows: rowsToEmbed.length,
    embeddingPrefilter: EMBEDDING_PREFILTER,
    rankingWeights,
    results,
  };
}

async function main() {
  const response = await runSearch(QUERY);
  console.log(JSON.stringify(response));
}

async function writeWorkerResponse(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function parseWorkerRequest(line) {
  const trimmed = line.trim();
  if (!trimmed) return null;

  try {
    const parsed = JSON.parse(trimmed);
    return String(parsed.query ?? parsed.prompt ?? '').trim();
  } catch {
    return trimmed;
  }
}

async function startStdioWorker() {
  process.stderr.write('Local AI search worker ready.\n');
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

  for await (const line of rl) {
    const query = parseWorkerRequest(line);
    if (!query) {
      await writeWorkerResponse({ error: 'Please provide a search prompt.' });
      continue;
    }

    try {
      const response = await runSearch(query);
      await writeWorkerResponse(response);
    } catch (error) {
      await writeWorkerResponse({
        error: 'Local AI search failed.',
        details: error?.stack || String(error),
      });
    }
  }
}

const entrypoint = STDIO_MODE ? startStdioWorker : main;
entrypoint().catch((error) => {
  fail('Local AI search failed.', error?.stack || String(error));
});
