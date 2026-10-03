import sharp from 'sharp';

const MAX_INPUT = 64 * 1024;
const MAX_SVG = 1024 * 1024;
const LIMITS = {width: 2048, height: 1024, pixels: 2097152, png: 1048576};

function fail(error) { return {ok: false, error}; }
function sizeInPx(value) {
  const match = /^([0-9]+(?:\.[0-9]+)?)(ex|em|px)$/.exec(value || '');
  if (!match) throw Object.assign(new Error('invalid dimensions'), {renderCategory: 'too_large'});
  return Number(match[1]) * {ex: 8, em: 16, px: 1}[match[2]];
}

async function render(expression, scale, theme) {
  if (typeof expression !== 'string' || !expression.trim() || expression.length > 2000) return fail('invalid_input');
  if (/\\(?:require|href|html|includegraphics|newcommand|def|gdef|let|input|write|read|openout)\b/i.test(expression)) return fail('unsupported');
  try {
    // Reuse modules and font data, but never parser, macro, or label state.
    MathJax.startup.input = MathJax.startup.getInputJax();
    MathJax.startup.document = MathJax.startup.getDocument();
    MathJax.startup.makeMethods();
    const node = await MathJax.tex2svgPromise(expression, {display: true, em: 16, ex: 8, containerWidth: 1280});
    const adaptor = MathJax.startup.adaptor;
    const svgNode = adaptor.firstChild(node);
    const baseWidth = sizeInPx(adaptor.getAttribute(svgNode, 'width'));
    const baseHeight = sizeInPx(adaptor.getAttribute(svgNode, 'height'));
    const viewBox = (adaptor.getAttribute(svgNode, 'viewBox') || '').trim().split(/\s+/).map(Number);
    if (viewBox.length !== 4 || !viewBox.every(Number.isFinite) || viewBox[2] <= 0 || viewBox[3] <= 0) return fail('too_large');
    const width = Math.ceil(baseWidth * scale);
    const height = Math.ceil(baseHeight * scale);
    if (!width || !height || width > LIMITS.width || height > LIMITS.height || width * height > LIMITS.pixels) return fail('too_large');
    adaptor.setAttribute(svgNode, 'width', `${baseWidth}px`);
    adaptor.setAttribute(svgNode, 'height', `${baseHeight}px`);
    let svg = adaptor.serializeXML(svgNode);
    if (!svg.includes('xmlns=')) svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ');
    const foreground = theme === 'dark' ? '#f6f7f9' : '#18202b';
    const background = theme === 'dark' ? '#20252c' : '#f8f9fb';
    svg = svg.replace('<svg ', `<svg color="${foreground}" `);
    if (Buffer.byteLength(svg) > MAX_SVG) return fail('too_large');
    const padding = 16;
    if (width + 2 * padding > LIMITS.width || height + 2 * padding > LIMITS.height || (width + 2 * padding) * (height + 2 * padding) > LIMITS.pixels) return fail('too_large');
    const png = await sharp(Buffer.from(svg), {density: 72 * scale, limitInputPixels: LIMITS.pixels})
      .flatten({background}).extend({top: padding, bottom: padding, left: padding, right: padding, background})
      .png().toBuffer();
    if (png.length > LIMITS.png) return fail('too_large');
    const metadata = await sharp(png).metadata();
    if (!metadata.width || !metadata.height || metadata.width > LIMITS.width || metadata.height > LIMITS.height || metadata.width * metadata.height > LIMITS.pixels) return fail('too_large');
    return {ok: true, png_base64: png.toString('base64'), width: metadata.width, height: metadata.height};
  } catch (error) {
    if (error.renderCategory) return fail(error.renderCategory);
    if (/pixel limit/i.test(String(error))) return fail('too_large');
    throw error;
  }
}

async function main() {
  const chunks = [];
  let length = 0;
  for await (const chunk of process.stdin) {
    length += chunk.length;
    if (length > MAX_INPUT) throw new Error('input too large');
    chunks.push(chunk);
  }
  const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  if (input.version !== 1 || !Array.isArray(input.expressions) || input.expressions.length > 4 ||
      !Number.isFinite(input.scale) || input.scale < 0.5 || input.scale > 3 ||
      !['light', 'dark'].includes(input.theme)) throw new Error('invalid request');
  global.MathJax = {
    loader: {
      paths: {mathjax: '@mathjax/src/bundle'},
      load: ['input/tex-base', '[tex]/ams', 'output/svg', 'adaptors/liteDOM'],
      require: file => {
        if (!file.startsWith('@mathjax/')) throw new Error('nonlocal module rejected');
        return import(file);
      }
    },
    // This option is spelled "Subtitutions" in the pinned MathJax API.
    tex: {packages: ['base', 'ams'], maxBuffer: 4096, maxMacros: 1000, maxTemplateSubtitutions: 1000, formatError: (_jax, error) => {
      error.renderCategory = 'syntax';
      throw error;
    }},
    svg: {fontCache: 'none'},
    output: {font: 'mathjax-newcm'}
  };
  await import('@mathjax/src/bundle/startup.js');
  await MathJax.startup.promise;
  try {
    const results = [];
    for (const expression of input.expressions) results.push(await render(expression, input.scale, input.theme));
    process.stdout.write(JSON.stringify({version: 1, results}));
  } finally {
    MathJax.done();
  }
}

main().catch(() => {
  process.stderr.write('renderer startup/protocol failure\n');
  process.exitCode = 1;
});
