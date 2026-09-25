"""fix2: remove the clipped-away content of artifacts/figures/fb_v2_hero_channel.pdf (panels (b), (c) and the
panel-(a) title row), so that the PDF of Fig. 1(b) carries no hidden text. The page size is kept, so the
trim in sections/02_model.tex (trim=0 3 268.6 13.5) is unchanged. Output: manuscript figures/fb_v2_hero_channel_a.pdf."""
import fitz, sys
src, out = sys.argv[1], sys.argv[2]
d = fitz.open(src); pg = d[0]
W, H = pg.rect.width, pg.rect.height
xcut, ytop = W - 268.6, 13.5          # kept region: x < 185.0, y > 13.5 (fitz coordinates, y down)
pg.add_redact_annot(fitz.Rect(xcut, 0, W, H), fill=False)
pg.add_redact_annot(fitz.Rect(0, 0, xcut, ytop), fill=False)
pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
d.save(out, garbage=4, deflate=True)
print(W, H, [s['text'] for b in fitz.open(out)[0].get_text('dict')['blocks'] for l in b.get('lines', []) for s in l['spans']])
