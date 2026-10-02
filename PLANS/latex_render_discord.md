# Implementation Plan: LaTeX Rendering in Discord Bot

This document provides a comprehensive, step-by-step implementation guide for building and integrating a LaTeX rendering feature into a Discord bot using TypeScript, `discord.js`, `mathjax`, and `sharp`.

---

## 1. Overview & Architecture

### 1.1 Goal
Allow Discord users to input LaTeX equations (via slash commands or message triggers) and receive a cleanly formatted, high-resolution PNG image rendered in the Discord channel.

### 1.2 Pipeline
```
Discord Slash Command (/latex <code> [scale] [brightness])
       │
       ▼
1. Discord Bot Interaction Handler (deferReply immediately)
       │
       ▼
2. LaTeX Preprocessing (wrap in color, math environment: \color{white} \begin{aligned} ... \end{aligned})
       │
       ▼
3. MathJax 3 Engine (TeX input ──► SVG markup string)
       │
       ▼
4. Sharp Rasterizer (SVG buffer ──► scale, adjust brightness ──► PNG buffer)
       │
       ▼
5. Discord Response (editReply with Embed + attachment://rendered.png)
```

---

## 2. Dependencies & Prerequisites

### 2.1 Package Requirements
Install required runtime libraries:
```bash
npm install discord.js mathjax sharp
npm install --save-dev @types/sharp @types/node typescript
```
*(Or equivalent using `pnpm` / `yarn`)*.

### 2.2 Native Module Notes
- `sharp` depends on `libvips`. It automatically downloads precompiled binaries for most platforms (Windows, macOS, Linux x64/arm64). If deploying in Docker/Linux containers, ensure standard glibc or musl dependencies are present.
- `mathjax` (version 3.x) runs headlessly in Node.js without needing a browser DOM.

---

## 3. TypeScript Ambient Type Declarations

MathJax v3 lacks complete ambient types for Node.js usage. Create a type declaration file to avoid TypeScript compilation errors.

### File: `types/mathjax.d.ts` (or `global.d.ts`)
```typescript
declare module 'mathjax' {
  export function init(options: {
    loader: { load: string[] };
    [key: string]: unknown;
  }): Promise<{
    tex2svgPromise: (
      tex: string,
      options?: { display?: boolean; [key: string]: unknown }
    ) => Promise<unknown>;
  }>;
}

declare namespace MathJax {
  const startup: {
    adaptor: {
      innerHTML: (node: unknown) => string;
    };
  };
}
```

---

## 4. Step-by-Step Implementation

### Step 1: Create the Dedicated LaTeX Rendering Service
Encapsulate the MathJax and Sharp rendering logic into a standalone service. This keeps command handlers clean and makes testing straightforward.

#### File: `src/services/latexRenderer.ts`
```typescript
import mathjax from 'mathjax';
import sharp from 'sharp';

export interface RenderOptions {
  scale?: number;       // Scaling factor (default: 2)
  brightness?: number;  // Brightness multiplier (0 to 1, default: 0.8)
  color?: string;       // Text color (default: 'white')
}

export class LatexRenderer {
  private static jaxInstance: any = null;
  private static initPromise: Promise<any> | null = null;

  /**
   * Initializes the MathJax singleton instance.
   */
  public static async getJax() {
    if (this.jaxInstance) return this.jaxInstance;
    if (!this.initPromise) {
      this.initPromise = mathjax.init({
        loader: {
          load: ['input/tex', 'output/svg'],
        },
      });
    }
    this.jaxInstance = await this.initPromise;
    return this.jaxInstance;
  }

  /**
   * Renders a LaTeX string to a PNG Buffer.
   *
   * @param code Raw LaTeX expression from user
   * @param options Styling and raster options
   * @returns PNG Buffer
   */
  public static async renderToPng(
    code: string,
    options: RenderOptions = {}
  ): Promise<Buffer> {
    const {
      scale = 2,
      brightness = 0.8,
      color = 'white',
    } = options;

    const jax = await this.getJax();

    // Wrap equation: set foreground color and wrap in an aligned environment
    const wrappedCode = `\\color{${color}} \\begin{aligned} ${code.trim()} \\end{aligned}`;

    // 1. TeX to SVG conversion
    const svgNode = await jax.tex2svgPromise(wrappedCode, { display: true });
    const svgString = MathJax.startup.adaptor.innerHTML(svgNode);

    // 2. Validate that valid SVG was produced
    if (!svgString || !svgString.includes('<svg')) {
      throw new Error('Failed to generate valid SVG from LaTeX input.');
    }

    // Check for MathJax internal parsing errors (renders <merror>)
    if (svgString.includes('data-mjx-error') || svgString.includes('<merror')) {
      throw new Error('LaTeX syntax error.');
    }

    // 3. Rasterize and process with Sharp
    const sharpInstance = sharp(Buffer.from(svgString));
    const metadata = await sharpInstance.metadata();
    const baseHeight = metadata.height || 32;

    const pngBuffer = await sharpInstance
      .resize({
        height: Math.ceil(baseHeight * Math.min(Math.max(scale, 0.5), 5)),
      })
      .modulate({
        brightness: Math.min(Math.max(brightness, 0.1), 1.0),
      })
      .png({ effort: 1 })
      .toBuffer();

    return pngBuffer;
  }
}
```

---

### Step 2: Implement the Slash Command

#### File: `src/commands/latex.ts`
```typescript
import {
  ChatInputCommandInteraction,
  SlashCommandBuilder,
  EmbedBuilder,
  AttachmentBuilder,
} from 'discord.js';
import { LatexRenderer } from '../services/latexRenderer.js';

export const data = new SlashCommandBuilder()
  .setName('latex')
  .setDescription('Render a LaTeX equation into an image')
  .addStringOption(option =>
    option
      .setName('code')
      .setDescription('LaTeX expression to render')
      .setRequired(true)
  )
  .addNumberOption(option =>
    option
      .setName('scale')
      .setDescription('Image scale multiplier (0.5 to 5, default 2)')
      .setMinValue(0.5)
      .setMaxValue(5)
  )
  .addNumberOption(option =>
    option
      .setName('brightness')
      .setDescription('Brightness adjustment (0.1 to 1.0, default 0.8)')
      .setMinValue(0.1)
      .setMaxValue(1)
  );

export async function execute(interaction: ChatInputCommandInteraction) {
  // CRITICAL: Defer immediately to satisfy Discord's 3-second acknowledgement window
  await interaction.deferReply();

  const code = interaction.options.getString('code', true);
  const scale = interaction.options.getNumber('scale') ?? 2;
  const brightness = interaction.options.getNumber('brightness') ?? 0.8;

  try {
    const pngBuffer = await LatexRenderer.renderToPng(code, {
      scale,
      brightness,
    });

    const attachment = new AttachmentBuilder(pngBuffer, {
      name: 'latex.png',
    });

    const embed = new EmbedBuilder()
      .setColor(0x2f3136)
      .setImage('attachment://latex.png')
      .setFooter({ text: 'Rendered with MathJax' });

    await interaction.editReply({
      embeds: [embed],
      files: [attachment],
    });
  } catch (error: any) {
    const isSyntaxError = error?.message?.includes('LaTeX syntax error');
    const userMessage = isSyntaxError
      ? '❌ **LaTeX Syntax Error**: Please check your TeX markup.'
      : '❌ **Error**: Failed to render LaTeX equation.';

    await interaction.editReply({
      content: userMessage,
    });
  }
}
```

---

### Step 3: Register Command in Bot Entry Point

Ensure commands are deployed to Discord API and wired to interaction listeners.

#### File: `src/index.ts` (Summary of essential wiring)
```typescript
import { Client, GatewayIntentBits, Events, REST, Routes } from 'discord.js';
import * as latexCommand from './commands/latex.js';

const client = new Client({
  intents: [GatewayIntentBits.Guilds],
});

// Register slash command on ready
client.once(Events.ClientReady, async c => {
  console.log(`Logged in as ${c.user.tag}`);

  const rest = new REST().setToken(process.env.DISCORD_TOKEN!);
  await rest.put(Routes.applicationCommands(c.user.id), {
    body: [latexCommand.data.toJSON()],
  });
  console.log('Slash commands registered.');
});

// Handle command execution
client.on(Events.InteractionCreate, async interaction => {
  if (!interaction.isChatInputCommand()) return;

  if (interaction.commandName === 'latex') {
    try {
      await latexCommand.execute(interaction);
    } catch (err) {
      console.error('Unhandled interaction error:', err);
      if (interaction.deferred || interaction.replied) {
        await interaction.editReply({ content: 'An unexpected error occurred.' });
      } else {
        await interaction.reply({ content: 'An unexpected error occurred.', ephemeral: true });
      }
    }
  }
});

client.login(process.env.DISCORD_TOKEN);
```

---

## 5. Critical Best Practices & Gotchas

| Challenge | Why it Happens | Recommended Solution |
| :--- | :--- | :--- |
| **Discord Interaction Timeout** | Discord requires acknowledgment within 3,000ms. MathJax parsing + Sharp rasterization can exceed this during cold start or high server load. | Always invoke `await interaction.deferReply()` immediately as the first line of the command execution. |
| **Unreadable Dark Mode Text** | Default TeX text color is `#000000` (black), which is invisible against Discord's dark theme backgrounds. | Prepend `\color{white}` or wrap in `<svg>` styling before rasterizing. Using a brightness factor around `0.8` prevents high-contrast eye strain. |
| **MathJax Singleton Initialization** | Calling `mathjax.init()` repeatedly per command introduces massive latency and memory consumption. | Initialize MathJax **once** at application launch and reuse the instance for all rendering requests. |
| **Invalid Syntax Crashing Process** | Malformed TeX inputs could throw or produce `<merror>` nodes in the SVG tree. | Inspect the generated SVG for `data-mjx-error` or `<merror>` tags and return a descriptive user-facing error message instead of failing silently. |
| **Multi-line Math Support** | Raw LaTeX standard commands do not support alignment across lines without math environments. | Wrap user input inside `\begin{aligned} ... \end{aligned}` so users can use `\\` and `&` formatting natively. |

---

## 6. Verification & Test Suite

Verify the implementation against the following test inputs:

1. **Basic Equation**:
   - Input: `E = mc^2`
   - Expected: Single line equation with white text.

2. **Fractions & Integrals**:
   - Input: `\int_{0}^{\infty} \frac{x^3}{e^x - 1} \, dx = \frac{\pi^4}{15}`
   - Expected: Full height fraction and integral symbols properly rendered.

3. **Multi-line Aligned System**:
   - Input: `a + b &= 10 \\ 2a - b &= 5`
   - Expected: Proper vertical alignment at the `&` symbol.

4. **Matrices**:
   - Input: `\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}`
   - Expected: 2x2 matrix with curved parentheses.

5. **Syntax Error Handling**:
   - Input: `\frac{1}{`
   - Expected: Bot catches syntax error and replies with user-friendly error notice, without crashing.
