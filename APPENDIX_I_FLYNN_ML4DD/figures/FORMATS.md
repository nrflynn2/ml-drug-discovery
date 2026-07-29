# figures/ format note

Every figure ships as PNG at 300 dpi, SVG and PDF, with one deliberate exception.

`boltz_showcase` has PNG and PDF only. It is a ray-traced PyMOL render, so it is a bitmap by nature;
an SVG would be that same bitmap wrapped in an XML envelope, larger on disk with nothing gained. The
PDF is the vector-container version to hand a typesetter.

The plotted figures are true vector SVGs, so text and lines stay sharp at any size.
