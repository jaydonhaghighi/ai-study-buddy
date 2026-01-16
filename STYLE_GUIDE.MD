
-----

# UI Design System: AI Study Platform

## 1\. Design Philosophy

  * **Keywords:** Intellectual, Tactile, Calm, Focused.
  * **Concept:** "Digital Parchment." The interface should feel like a fresh notebook. It avoids the coldness of standard tech software by using warm neutrals and texture, creating a comfortable environment for long study sessions.

-----

## 2\. Color Palette (Strictly Neutral)

To maintain the professional minimalism, we remove all chromatic distractions. We rely on value (light vs. dark) to create hierarchy.

### **The Foundation (Backgrounds)**

  * **Paper White:** `#F9F9F7` (The main canvas. Not pure white, slightly warm.)
  * **Mist Grey:** `#EFEFE9` (Secondary backgrounds, sidebars, active states.)
  * **Overlay White:** `#FFFFFF` (Used sparingly for inputs or cards that need to lift off the page.)

### **The Ink (Typography & Elements)**

  * **Charcoal (Primary Text):** `#222222` (High contrast but softer than pure black.)
  * **Graphite (Secondary Text):** `#555555` (Subtitles, metadata.)
  * **Stone (Borders/Dividers):** `#D1D1D1` (Subtle structure.)
  * **Silver (Disabled/Placeholders):** `#A0A0A0`

### **Semantic Neutrals (Status)**

  * **Success:** `#4A4A4A` (Solid dark grey badge)
  * **Error:** `#000000` (Thick borders or underline)
  * **Focus Ring:** `#888888` (Double border line)

-----

## 3\. Atmospheric Texture (The Paper Effect)

To achieve the "graininess," we will not use a static image for the whole background. Instead, we use a CSS overlay. This ensures the text remains crisp while the background feels textured.

**Implementation Strategy:**

> Apply a fixed position `div` over the background with a noise SVG or high-ISO noise image, set opacity to 3-5%, and use `pointer-events: none`.

**CSS Snippet:**

```css
.paper-texture {
  background-color: #F9F9F7;
  position: relative;
}

.paper-texture::before {
  content: "";
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  opacity: 0.05; /* Subtle grain */
  background-image: url("path/to/noise.png");
  pointer-events: none;
  z-index: 0;
}
```

-----

## 4\. Typography

This pairing creates a classic "Textbook" feel updated for the web.

### **Headings: Noto Serif**

  * **Usage:** H1, H2, H3, Hero text.
  * **Weight:** Semi-Bold (600) or Bold (700).
  * **Letter Spacing:** -0.02em (Tighter for a modern editorial look).
  * **Example Styles:**
      * **H1:** 2.5rem (40px) | *Charcoal*
      * **H2:** 2.0rem (32px) | *Charcoal*
      * **H3:** 1.5rem (24px) | *Graphite*

### **Body: Noto Sans**

  * **Usage:** Paragraphs, UI elements, Buttons, Chat bubbles.
  * **Weight:** Regular (400) and Medium (500).
  * **Line Height:** 1.6 (Generous spacing for readability during studying).
  * **Example Styles:**
      * **Body:** 1rem (16px) | *Charcoal*
      * **Small/Label:** 0.875rem (14px) | *Graphite* | Uppercase tracking 0.05em.

-----

## 5\. UI Component Library

### **Buttons**

Minimalist buttons that look like ink stamps or sketches.

  * **Primary Button:**
      * Background: `#222222` (Charcoal)
      * Text: `#FFFFFF` (White) - Noto Sans Medium
      * Radius: `4px` (Slightly rounded, avoiding the "pill" shape).
      * Hover: slight opacity drop (90%).
  * **Secondary Button:**
      * Background: Transparent
      * Border: `1px solid #D1D1D1`
      * Text: `#222222`
      * Hover: Background `#EFEFE9`.

### **Inputs & Text Areas (The "Blank Line" feel)**

  * **Style:** Instead of a full box, use a bottom border only *OR* a very pale background fill with no border.
  * **Default:** Background `#FFFFFF` with Border `#D1D1D1` (1px).
  * **Focus:** Border becomes `#222222` (2px). No blue glow.

### **The Chat Interface (Study Buddy)**

Since we have no accent colors, we distinguish the AI from the User using background values and alignment.

  * **AI Response Bubble:**
      * Align: Left
      * Background: Transparent (blends into paper texture).
      * Border: Left border line `2px solid #D1D1D1`.
      * Font: Noto Serif (To distinguish the "Expert" knowledge).
  * **User Prompt Bubble:**
      * Align: Right
      * Background: `#EFEFE9` (Mist Grey).
      * Radius: `8px`.
      * Font: Noto Sans.

### **Cards / Containers**

Avoid drop shadows to keep the interface flat and clean. Use borders.

  * **Border:** `1px solid #E5E5E5`.
  * **Background:** `#FFFFFF` (Overlay White).
  * **Padding:** Generous (24px+).

-----

## 6\. Layout & Spacing

  * **Grid:** 8pt grid system.
  * **Whitespace:** High density. Since the design is minimal, whitespace is the primary separator.
      * *Margins:* Small elements separated by 16px. Sections separated by 64px or 80px.
  * **Max Width:** Text columns (for study materials) should not exceed `65ch` (65 characters) wide to ensure comfortable reading.

-----
