# Brush Fill for grease pencil
Blender addon - Grease pencil flat brush that act like a 2D brush for fill materials

**[Download latest](https://github.com/Pullusb/GP_brush_fill/archive/master.zip)**

/!\ Need external modules (see section below [_How to get opencv and shapely modules_](#how-to-get-opencv-and-shapely-modules))

---  

## Description

Paint Grease pencil closed flat stroke with a 2D brush on a plane defined by same projecting option as usual grease pencil.
Projection modes "*Surface*" and "*Stroke*" are not supported (act as if *Origin* is selected).

This addon is to be considered as a workaround until better native blender solution exists for flat coloring.

### Usage

The UI panel appears once a grease pencil object is selected, in the left toolbar.
The brush fill modal operator is lauched with a button `brush fill`.
You can bind a shortcut to it by right clicking on the brush fill button and sue `add shortcut`
In the addon preferences you can enable an option to bind a default `ctrl+shit+F` shortcut  

`click` : Normal paint, add grease pencil stroke with current selected materials.

Combining with shift/alt before releasing allow add/sub modes with existing grease pencil strokes.
This modes have filter available in user interface (filter what types of grease pencil strokes to affect)

`shift`+`click` : Add, paint in additive mode. merge crossed strokes

`alt` + `click` : Eraser, paint in substractive mode. merge crossed strokes too

### Limitations

Everything happens in the screen space visible part of the viewport that is locked during the modal.
Meaning that you can't move during the pain and the brush is clamped to viewport border.

### Important technical note

This addon rely on external python modules :

**opencv** to use the "paint" in the region screen pixel with circle brush and convert it polygon > grease pencil strokes

**shapely** for additive/substractive boolean operation between polygon generated by the prush and existing grease pencil strokes.

<!-- All created grease pencil strokes have *draw_cyclic* option set to *True* and last point isn't superposed to first -->

## How to get opencv and shapely modules

Note : the package name for opencv is `opencv-python`, or `opencv-contrib-python` for extended version
the module folder will be named `cv2`

You need to have python installed on your machine to install with Pip:
[Detailed infos on module installation here](https://docs.python.org/3/installing/index.html)
The local python version need to be the same version as blender python : `3.7`.
To check pip python version enter : `pip --version` or `pip -V`
If your version is ok, run this lines and go to the _Place modules in blender_ section:
`pip install opencv-contrib-python`
`pip install shapely`

If you have multiple python version installed you can choose like this :
On linux `python3.7 -m pip install shapely` (if you seem to have a permission problem add `--user` to this line to install for current user only)
On windows `py -3.7 -m pip install shapely`

Details on packages here :
opencv: https://pypi.org/project/opencv-python/
shapely : https://pypi.org/project/Shapely/

For windows user there is the alternative solution of the [_Unofficial Windows Binaries for Python Extension Packages_](https://www.lfd.uci.edu/~gohlke/pythonlibs/).
Download `whl` of [opencv](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv) and [shapely](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely). In the list choose the last version that is compatible with your system. `amd64` for 64bits else get `win 32`.
Then open a terminal in the folder containing these file and run `pip install ThePackage.whl` ([more info on Wheel installation](https://pip.pypa.io/en/latest/user_guide/#installing-from-wheels))


### Locate modules

Once installed you need to copy module files from your local python into your blender module folder.
Those file are stored in the python version directory "site-packages".
To easily locate them use : `pip show shapely`, that will display their location.
ex: On windows it will probably be something like  `C:/Users/User/AppData/Local/Programs/Python/Python-version/Lib/site-packages`

### Place modules in blender

Now copy the folder `cv2` and `shapely` in you blender module folder
Don't mind the folders ending with `.dist-info`
Blender module folder is in temp directory and (there is another in install directory).
To locate it open blender and in the python interactive console and run `print(bpy.utils.user_resource('SCRIPTS'))` (or use my [devtool](https://github.com/Pullusb/devTools) addon button "print resources filepath" ;) )
Copy the modules in the `modules` folder inside `scripts` (if theres no "modules" folder just create it).
The path would be something like `.../Blender/2.80/scripts/modules/`

Once modules are in this folder you can use them right away (no need to restart blender). Just launch the operator. \o/

---

#### Todo:
- add a curve pressure for pen

- add thickness as a scene properties (exposed to UI if any)

- add specific material path to use (automaticly added to material of the object if not listed)

- maybe : add a way to handle rapid radius change (like 'F' for classic blender brush.)

- in additive mode : add a shape when nothing is instersected

- maybe reorder encountered shape list by screen space proximity to stroke starting point (might greatly slow down)

- Handle holes in shapes (detected Hole shapes may goes on a substractive layer on top OR two shapes side by side)
    - need to find best naming convention and handling method...

- find relative brush size and spacing according to region resolution / pixel density

- re-sample shape to get uniform contour points spacing along line (quite hard)
    - Can be use on single line but in that case must manage to keep point properties (and interpolate them when creating in-betweens)...

<!-- - Stanby because not reliable with filled shape : handle Surface project mode (raycast on first point to underlying object) -->

- re-enable coplanarity check:
    - Fix bug in coplanar check (sometimes not evaluated as coplanar even)
    - maybe divide in two function or condition to check without object matrix applyed.

- coplanar handling:
    what to do when multiple strokes have coplanar points but not colanar between them ?
    need project from view (on centroid or on selected point)
    need self make coplanar with average normal and point position.
    might need an option to "straighten" a stroke, rotate it on centroid (not project) so it fit chosen object axis
    very optional: might be cool quantify "coplanarity" to allow coplanar-tolerance contitions.
    very optional: align view to selected stroke normal


#### Todo UI:
- Display thickness

- material selector (to specidy what material is gonna be use for the paint)
    - material selector UI
        select a material that use the fill (if not selected)
        OR select a predefined brush (that will have a material locked)
        but there will be different fill.
        might be cool to create a "material selector" field in UI later.
        generate stroke (what to do with pressure ?...)

#### Done:
- test for more precision : Upscale the points coordinates and the empty numpy array before tracing the the circles in the numpy array, then trace and downscale.
- omit final point in shape on gp stroke creation.
- add pressure control for tablets

- add/substract (erase) stroke - (shapely)
    - handle when shape is contained (and not intersected)
    - Fix shape when substracted on multiple shape simultaneously.
    - add angle warning
- stop without error if not enough element in mouse path list

- reproject stroke on axis plane (if axis locked) or with intersected stroke (first ?)
    - respect stroke placement

- show operation type (+, -, x) if shift, alt

<!-- notes:
    #How to draw
    settings = bpy.context.scene.tool_settings

    #Drawing plane : Drawplane orientation (normal)
    settings.gpencil_sculpt.lock_axis = 'VIEW'
    settings.gpencil_sculpt.lock_axis = 'AXIS_Y'# front (X-Z)
    settings.gpencil_sculpt.lock_axis = 'AXIS_X'# side (Y-Z)
    settings.gpencil_sculpt.lock_axis = 'AXIS_Z'# top (X-Y)
    settings.gpencil_sculpt.lock_axis = 'CURSOR'

    #Stroke placement : "location" (depth)
    settings.gpencil_stroke_placement_view3d = 'ORIGIN'
    settings.gpencil_stroke_placement_view3d = 'CURSOR'
    settings.gpencil_stroke_placement_view3d = 'SURFACE'
    settings.gpencil_stroke_placement_view3d = 'STROKE' -->