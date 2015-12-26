# -*- coding: utf-8 -*-
#
# Experiments with Schemes for Exponential Decay documentation build configuration file, created by
# sphinx-quickstart on Sun Dec 20 03:39:25 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
          #'sphinx.ext.pngmath',
          #'sphinx.ext.jsmath',
          'sphinx.ext.mathjax',
          #'matplotlib.sphinxext.mathmpl',
          #'matplotlib.sphinxext.only_directives',
          'matplotlib.sphinxext.plot_directive',
          'sphinx.ext.autodoc',
          'sphinx.ext.doctest',
          'sphinx.ext.viewcode',
          'sphinx.ext.intersphinx',
          'sphinx.ext.inheritance_diagram',
          'IPython.sphinxext.ipython_console_highlighting']

#pngmath_dvipng_args = ['-D 200', '-bg Transparent', '-gamma 1.5']  # large math fonts (200)

# Make sphinx aware of the DocOnce lexer
def setup(app):
    from sphinx.highlighting import lexers
    from doconce.misc import DocOnceLexer
    lexers['doconce'] = DocOnceLexer()

# Check which additional themes that are installed
additional_themes_installed = []
additional_themes_url = {}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['_themes']
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

try:
    import alabaster
    additional_themes_installed.append('alabaster')
except ImportError:
    additional_themes_url['alabaster'] = 'sudo pip install alabaster'

try:
    import sphinxjp.themes.solarized
    additional_themes_installed.append('solarized')
except ImportError:
    additional_themes_url['solarized'] = 'https://bitbucket.org/miiton/sphinxjp.themes.solarized: sudo pip install -e hg+https://bitbucket.org/miiton/sphinxjp.themes.solarized#egg=sphinxjp.themes.solarized --upgrade'

try:
    import cloud_sptheme as csp
    additional_themes_installed.append('cloud')
    additional_themes_installed.append('redcloud')
except ImportError:
    url = 'https://bitbucket.org/ecollins/cloud_sptheme: sudo pip install -e hg+https://bitbucket.org/ecollins/cloud_sptheme#egg=cloud_sptheme --upgrade'
    additional_themes_url['cloud'] = url
    additional_themes_url['redcloud'] = url


'''
# FIXME: think we do not need to test on basicstrap, but some themes
# need themecore and we must test for that
try:
    import sphinxjp.themecore
    if not 'sphinxjp.themecore' in extensions:
        extensions += ['sphinxjp.themecore']
    additional_themes_installed.append('basicstrap')
except ImportError:
    # Use basicstrap as an example on a theme with sphinxjp.themecore (??)
    additional_themes_url['basicstrap'] = 'https://github.com/tell-k/sphinxjp.themes.basicstrap: sudo pip install -e git+https://github.com/ryan-roemer/sphinx-bootstrap-theme#egg=sphinx-bootstrap-theme --upgrade'
'''

try:
    import sphinxjp.themes.impressjs
    additional_themes_installed.append('impressjs')
except ImportError:
    additional_themes_url['impressjs'] = 'https://github.com/shkumagai/sphinxjp.themes.impressjs: sudo pip install -e git+https://github.com/shkumagai/sphinxjp.themes.impressjs#egg=sphinxjp.themes.impressjs --upgrade'

try:
    import sphinx_bootstrap_theme
    additional_themes_installed.append('bootstrap')
except ImportError:
    additional_themes_url['bootstrap'] = 'https://github.com/ryan-roemer/sphinx-bootstrap-theme: sudo pip install -e git+https://github.com/ryan-roemer/sphinx-bootstrap-theme#egg=sphinx-bootstrap-theme --upgrade'

try:
    import icsecontrib.sagecellserver
    extensions += ['icsecontrib.sagecellserver']
except ImportError:
    # sudo pip install -e git+https://github.com/kriskda/sphinx-sagecell#egg=sphinx-sagecell --upgrade
    pass

# Is the document built on readthedocs.org? If so, don't import
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:  # only import and set the theme if we're building docs locally
    try:
        import sphinx_rtd_theme
        additional_themes_installed.append('sphinx_rtd_theme')
    except ImportError:
        additional_themes_url['sphinx_rtd_theme'] = 'sudo pip install sphinx_rtd_theme'

tinker_themes = [
  'dark', 'flat', 'modern5', 'minimal5', 'responsive']
# http://tinkerer.me/index.html
# See Preview Another Theme in the sidebar of the above URL
try:
    import tinkerer
    import tinkerer.paths
    additional_themes_installed += tinker_themes
except ImportError:
    for theme in tinker_themes:
        additional_themes_url[theme] = 'sudo pip install tinkerer --upgrade'



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Experiments with Schemes for Exponential Decay'
short_title = project
copyright = u'2015, Hans Petter Langtangen. Released under CC Attribution 4.0 license'
author = u'Hans Petter Langtangen'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '1.0'
# The full version, including alpha/beta/rc tags.
release = '1.0'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'responsive'
#html_theme = 'ADCtheme'
#html_theme = 'agni'
#html_theme = 'agogo'
#html_theme = 'alabaster'
#html_theme = 'basic'
#html_theme = 'basicstrap'
#html_theme = 'bizstyle'
#html_theme = 'bloodish'
#html_theme = 'boilerplate'
#html_theme = 'bootstrap'
#html_theme = 'cbc'
#html_theme = 'classic'
#html_theme = 'cloud'
#html_theme = 'dark'
#html_theme = 'default'
#html_theme = 'epub'
#html_theme = 'fenics'
#html_theme = 'fenics_minimal1'
#html_theme = 'fenics_minimal2'
#html_theme = 'flat'
#html_theme = 'haiku'
#html_theme = 'jal'
#html_theme = 'minimal5'
#html_theme = 'modern5'
#html_theme = 'nature'
#html_theme = 'pylons'
#html_theme = 'pyramid'
#html_theme = 'redcloud'
#html_theme = 'responsive'
#html_theme = 'scipy_lectures'
#html_theme = 'scrolls'
#html_theme = 'slim-agogo'
#html_theme = 'solarized'
#html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinxdoc'
#html_theme = 'traditional'
#html_theme = 'uio'
#html_theme = 'vlinux-theme'

check_additional_themes = [
   'solarized', 'cloud', 'redcloud',
   'alabaster', 'bootstrap', 'impressjs']

for theme in check_additional_themes:
    if html_theme == theme:
        if not theme in additional_themes_installed:
            raise ImportError(
                'html_theme = "%s", but this theme is not installed. %s' % (theme, additional_themes_url[theme]))

if html_theme == 'solarized':
    pygments_style = 'solarized'



# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# See http://sphinx.pocoo.org/theming.html for options
if html_theme in ('default', 'classic'):
    # pygments_style =
    html_theme_options = {
       'rightsidebar': 'false',  # 'true'
       'stickysidebar': 'false', # Make the sidebar "fixed" so that it doesn't scroll out of view for long body content.  This may not work well with all browsers.  Defaults to false.
       'collapsiblesidebar': 'false', # Add an *experimental* JavaScript snippet that makes the sidebar collapsible via a button on its side. *Doesn't work together with "rightsidebar" or "stickysidebar".* Defaults to false.
       'externalrefs': 'false', # Display external links differently from internal links.  Defaults to false.
       # For colors and fonts, see default/theme.conf for default values
       #'footerbgcolor':    # Background color for the footer line.
       #'footertextcolor:'  # Text color for the footer line.
       #'sidebarbgcolor':   # Background color for the sidebar.
       #'sidebarbtncolor':  # Background color for the sidebar collapse button (used when *collapsiblesidebar* is true).
       #'sidebartextcolor': # Text color for the sidebar.
       #'sidebarlinkcolor': # Link color for the sidebar.
       #'relbarbgcolor':    # Background color for the relation bar.
       #'relbartextcolor':  # Text color for the relation bar.
       #'relbarlinkcolor':  # Link color for the relation bar.
       #'bgcolor':          # Body background color.
       #'textcolor':        # Body text color.
       #'linkcolor':        # Body link color.
       #'visitedlinkcolor': # Body color for visited links.
       #'headbgcolor':      # Background color for headings.
       #'headtextcolor':    # Text color for headings.
       #'headlinkcolor':    # Link color for headings.
       #'codebgcolor':      # Background color for code blocks.
       #'codetextcolor':    # Default text color for code blocks, if not set differently by the highlighting style.
       #'bodyfont':         # Font for normal text.
       #'headfont':         # Font for headings.
    }

elif html_theme == 'alabaster':
    # Doc: https://pypi.python.org/pypi/alabaster
    extensions += ['alabaster']
    html_theme_path += [alabaster.get_path()]
    html_theme_sidebars = {
      '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
      ]
    }

elif html_theme == 'sphinx_rtd_theme':
    # Doc: https://pypi.python.org/pypi/sphinx_rtd_theme
    if not on_rtd:
        html_theme_path += [sphinx_rtd_theme.get_html_theme_path()]

elif html_theme == 'sphinxdoc':
    # Doc: http://sphinx-doc.org/theming.html
    html_theme_options = {
       'nosidebar': 'false',  # 'true'
    }

elif html_theme == 'solarized':
    extensions += ['sphinxjp.themecore', 'sphinxjp.themes.solarized']

elif html_theme in ('cloud', 'redcloud'):
    html_theme_path += [csp.get_theme_dir()]

elif html_theme == 'impressjs':
    html_theme_path += [csp.get_theme_dir()]
    if not 'sphinxjp.themecore' in extensions:
        extensions += ['sphinxjp.themecore']

elif html_theme == 'scrolls':
    # Doc: http://sphinx.pocoo.org/theming.html
    pass
    #html_theme_options = {
       #'headerbordercolor':,
       #'subheadlinecolor:',
       #'linkcolor':,
       #'visitedlinkcolor':
       #'admonitioncolor':
    #}

elif html_theme == 'agogo':
    # Doc: http://sphinx.pocoo.org/theming.html
    pass

elif html_theme == 'nature':
    # Doc: http://sphinx.pocoo.org/theming.html
    html_theme_options = {
       'nosidebar': 'false',  # 'true'
    }

elif html_theme == 'traditional':
    # Doc: http://sphinx.pocoo.org/theming.html
    html_theme_options = {
       'nosidebar': 'false',  # 'true'
    }

elif html_theme == 'haiku':
    # Doc: http://sphinx.pocoo.org/theming.html
    html_theme_options = {
       'nosidebar': 'false',  # 'true'
    }

elif html_theme == 'pyramid':
    # Doc: http://sphinx.pocoo.org/theming.html
    html_theme_options = {
       'nosidebar': 'false',  # 'true'
    }

elif html_theme == 'bizstyle':
    # Doc: http://sphinx.pocoo.org/theming.html
    html_theme_options = {
       'nosidebar': 'false',  # 'true'
       'rightsidebar': 'false',  # 'true'
    }

elif html_theme == 'epub':
    # Doc: http://sphinx.pocoo.org/theming.html
    html_theme_options = {
       'relbar1': 'true',
       'footer': 'true',
    }

elif html_theme == 'basicstrap':
    html_theme_options = {
       'rightsidebar': 'false',  # 'true'
    }

elif html_theme == 'bootstrap':
    # Doc: https://ryan-roemer.github.io/sphinx-bootstrap-theme/README.html#customization
    html_theme_options = {
        'navbar_title': short_title,

        # Global TOC depth for "site" navbar tab. (Default: 1)
        # Switching to -1 shows all levels.
        'globaltoc_depth': -1,

        # HTML navbar class (Default: "navbar") to attach to <div> element.
        # For black navbar, do "navbar navbar-inverse"
        'navbar_class': "navbar navbar-inverse",

        # Fix navigation bar to top of page?
        # Values: "true" (default) or "false"
        'navbar_fixed_top': "true",

        # Location of link to source.
        # Options are "nav" (default), "footer" or anything else to exclude.
        'source_link_position': "footer",

        # Any Bootswatch theme (http://bootswatch.com/) can be used
        #'bootswatch_theme': 'readable',

        # A list of tuples containing pages or urls to link to.
        # Valid tuples should be in the following forms:
        #    (name, page)                 # a link to a page
        #    (name, "/aa/bb", 1)          # a link to an arbitrary relative url
        #    (name, "http://example.com", True) # arbitrary absolute url
        # Note the "1" or "True" value above as the third argument to indicate
        # an arbitrary url.
        #'navbar_links': [('PDF', '../mydoc.pdf', True), ('HTML', '../mydoc.html', True)],

        # TODO: Future.
        # Add page navigation to it's own navigation bar.
        #'navbar_page_separate': True,
    }
    html_theme_path += sphinx_bootstrap_theme.get_html_theme_path()

elif html_theme == 'scipy_lectures':
    # inherits the default theme and has all those options
    # set rightsidebar to true and nodesidebar to true to get
    # sidebar with the matching colors
    html_theme_options = {
        'nosidebar': 'true',
        'rightsidebar': 'false',
        'sidebarbgcolor': '#f2f2f2',
        'sidebartextcolor': '#20435c',
        'sidebarlinkcolor': '#20435c',
        'footerbgcolor': '#000000',
        'relbarbgcolor': '#000000',
    }

elif html_theme == 'cbc':
    pygments_style = "friendly"
elif html_theme == 'uio':
    pygments_style = "tango"
elif html_theme in tinker_themes:
    html_theme_options = {}
    extensions += ['tinkerer.ext.blog', 'tinkerer.ext.disqus']
    html_static_path += [tinkerer.paths.static]
    html_theme_path += [tinkerer.paths.themes]






# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = project

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = short_title

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.

if html_theme == 'impressjs':
    html_use_index = False


# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'ExperimentswithSchemesforExponentialDecaydoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',

# Latex figure (float) alignment
#'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'ExperimentswithSchemesforExponentialDecay.tex', u'Experiments with Schemes for Exponential Decay Documentation',
   u'Hans Petter Langtangen', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'experimentswithschemesforexponentialdecay', u'Experiments with Schemes for Exponential Decay Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'ExperimentswithSchemesforExponentialDecay', u'Experiments with Schemes for Exponential Decay Documentation',
   author, 'ExperimentswithSchemesforExponentialDecay', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False


# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The basename for the epub file. It defaults to the project name.
#epub_basename = project

# The HTML theme for the epub output. Since the default themes are not optimized
# for small screen space, using the same theme for HTML and epub output is
# usually not wise. This defaults to 'epub', a theme designed to save visual
# space.
#epub_theme = 'epub'

# The language of the text. It defaults to the language option
# or 'en' if the language is not set.
#epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
#epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#epub_identifier = ''

# A unique identification for the text.
#epub_uid = ''

# A tuple containing the cover image and cover page html template filenames.
#epub_cover = ()

# A sequence of (type, uri, title) tuples for the guide element of content.opf.
#epub_guide = ()

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
#epub_pre_files = []

# HTML files shat should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
#epub_post_files = []

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# The depth of the table of contents in toc.ncx.
#epub_tocdepth = 3

# Allow duplicate toc entries.
#epub_tocdup = True

# Choose between 'default' and 'includehidden'.
#epub_tocscope = 'default'

# Fix unsupported image types using the Pillow.
#epub_fix_images = False

# Scale large images.
#epub_max_image_width = 0

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#epub_show_urls = 'inline'

# If false, no index is generated.
#epub_use_index = True
