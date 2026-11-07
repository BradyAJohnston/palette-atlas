// Writes each CarbonPlan colormap, in light and dark mode, to
// src/node_palettes/data/rgb/carbonplan/<name>_<mode>.rgb as 255 colours, or to
// <name>.rgb alone when both modes are the same, replacing what was there.
//
// CarbonPlan's own scripts/export.mjs passes the mode as a positional argument,
// which makeColormap ignores, so every map it exports is the dark one. Here the
// options are passed as makeColormap expects them.
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { colormaps, makeColormap } from '@carbonplan/colormaps'

const COUNT = 255
const out = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../src/node_palettes/data/rgb/carbonplan'
)

const table = (name, mode) =>
  makeColormap(name, { mode, count: COUNT, format: 'rgb' })
    .map((c) => c.map((v) => Math.round(v)).join(' '))

const write = (file, rows) =>
  fs.writeFileSync(path.join(out, file), [`ncolors=${COUNT}`, ...rows, ''].join('\n'))

for (const file of fs.readdirSync(out)) {
  if (file.endsWith('.rgb')) fs.unlinkSync(path.join(out, file))
}
let written = 0
for (const { name } of colormaps) {
  const light = table(name, 'light')
  const dark = table(name, 'dark')
  if (light.join('\n') === dark.join('\n')) {
    write(`${name}.rgb`, light)
    written += 1
  } else {
    write(`${name}_light.rgb`, light)
    write(`${name}_dark.rgb`, dark)
    written += 2
  }
}
console.log(`wrote ${written} tables to ${out}`)
