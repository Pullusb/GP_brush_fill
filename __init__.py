# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
"name": "Brush fill",
"description": "Add a brush to paint flat grease pencil fills and add/erase existing strokes",
"author": "Samuel Bernou",
"version": (0, 2, 0),
"blender": (2, 80, 0),
"location": "Select a grease pencil object > 3D view > toolbar > brush fill (+shift for add, +alt for erase)",
"warning": "This addon need modules opencv and shapely to work",
"wiki_url": "https://github.com/Pullusb/GP_brush_fill",
"category": "3D View"
}

# coding: utf-8
import bpy
from mathutils import Vector, Matrix
from mathutils import geometry
import math
from math import sqrt
from time import time

import numpy as np
import gpu
import bgl
import blf

from time import time
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from gpu_extras.presets import draw_circle_2d

from bpy.types import Operator

# External module import
import cv2#openCV
import shapely### shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, MultiPolygon
from shapely.ops import split, cascaded_union


# -----------------
### 2D <> 3D
# -----------------


def location_to_region(worldcoords):
    ''' return 2d location '''
    from bpy_extras import view3d_utils
    return view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.space_data.region_3d, worldcoords)

def region_to_location(viewcoords, depthcoords):
    ''' return normalized 3d vector '''
    from bpy_extras import view3d_utils
    return view3d_utils.region_2d_to_location_3d(bpy.context.region, bpy.context.space_data.region_3d, viewcoords, depthcoords)

def get_view_origin_position():
    #method 1
    from bpy_extras import view3d_utils
    region = bpy.context.region
    rv3d = bpy.context.region_data
    view_loc = view3d_utils.region_2d_to_origin_3d(region, rv3d, (region.width/2.0, region.height/2.0))
    # print("view_loc1", view_loc)#Dbg
    
    """ #method 2
    r3d = bpy.context.space_data.region_3d
    view_loc2 = r3d.view_matrix.inverted().translation
    # print("view_loc2", view_loc2)#Dbg
    if view_loc != view_loc2: print('there might be an errror when finding view coordinate')
    """

    return view_loc

# -----------------
### Vector utils 2d
# -----------------

def vector_length_2d(A,B):
    ''''take two Vector and return length'''
    return sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)


def point_from_dist_in_segment_2d(a, b, dist, seglenght):
    '''return the tuple coords of a point on segment ab according to given distance and total segment lenght '''
    ## ref:https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    ratio = dist / seglenght
    #float : coord = ( ((1 - ratio) * a[0] + (ratio*b[0])), ((1 - ratio) * a[1] + (ratio*b[1])) )
    #mode int (for opencv function)
    coord = ( int( ((1 - ratio) * a[0] + (ratio*b[0])) ), int( ((1 - ratio) * a[1] + (ratio*b[1])) ) )
    return coord

def cross_vector_coord_2d(foo, bar, size):
    '''Return the coord in space of a cross vector between the two point with specified size'''
    ###middle location between 2 vector is calculated by adding the two vector and divide by two
    ##mid = (foo + bar) / 2
    between = foo - bar
    #create a generic Up vector (on Y or Z)
    up = Vector([0,1.0])
    new = Vector.cross(up, between)#the cross product return a 90 degree Vector
    if new == Vector([0.0000, 0.0000]):
        #new == 0 if up vector and between are aligned ! (so change up vector)
        up = Vector([0,-1.0,0])
        new = Vector.cross(up, between)#the cross product return a 90 degree Vector

    perpendicular = foo + new
    coeff = vector_length_coeff(size, foo, perpendicular)
    #position the point in space by adding the new vector multiplied by coeff value to get wanted lenght
    return (foo + (new * coeff))

def align_obj_to_vec(obj, v):
    '''align rotation to given vector'''
    up_axis = Vector([0.0, 0.0, 1.0])
    angle = v.angle(up_axis, 0)
    axis = up_axis.cross(v)
    euler = Matrix.Rotation(angle, 4, axis).to_euler()
    obj.rotation_euler = euler

def debug_create_empty_from_vec(p,v):
    new_obj = bpy.data.objects.new('new_obj', None)
    new_obj.empty_display_type = 'ARROWS'
    new_obj.empty_display_size = 2#0.5
    new_obj.location = p
    align_obj_to_vec(new_obj, v)
    bpy.context.scene.collection.objects.link(new_obj)

def midpoint_2d(p1, p2):
    return (Vector([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]))

def get_normal_from_tri(a,b,c):
    ab = b-a
    ac = c-a
    plane_no = ab.cross(ac)#.normalized()#not needed.
    return plane_no


def stroke_is_coplanar(s, tol=0.0002):
    '''
    Get a GP stroke object and tell if all points are coplanar (with a tolerance).
    return normal vector if coplanar else None
    '''

    if len(s.points) < 4:
        print('less than 4 points')
        return None#less than 4 points is necessaryly coplanar but not "evaluable" so retrun False

    obj = bpy.context.object
    mat = obj.matrix_world
    pct = len(s.points)
    a = mat @ s.points[0].co
    b = mat @ s.points[pct//3].co
    c = mat @ s.points[pct//3*2].co

    """ a = s.points[0].co
    b = s.points[1].co
    c = s.points[-2].co """
    ab = b-a
    ac = c-a

    #get normal (perpendicular Vector)
    plane_no = ab.cross(ac)#.normalized()
    val = plane_no

    # print('plane_no: ', plane_no)
    for i, p in enumerate(s.points):
        #let a tolerance value of at least 0.0002 maybe more
        """if abs(geometry.distance_point_to_plane(p.co, a, plane_no)) > tol:"""
        if abs(geometry.distance_point_to_plane(mat @ p.co, a, plane_no)) > tol:
            print('point', i, 'is not co-planar')
            print(i, geometry.distance_point_to_plane(p.co, a, plane_no))
            return False
            # val = None
    return val

# -----------------
### Utils functions
# -----------------

def get_addon_prefs():
    '''function to read current addon preferences properties'''
    import os
    addon_name = os.path.splitext(__name__)[0]
    preferences = bpy.context.preferences
    addon_prefs = preferences.addons[addon_name].preferences
    return (addon_prefs)

def transfer_value(Value, OldMin, OldMax, NewMin, NewMax):
    return (((Value - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

# -----------------
### Draw functions
# -----------------

def circle_2d(coord, r, num_segments):
    '''create circle, ref: http://slabode.exofire.net/circle_draw.shtml'''
    cx, cy = coord
    points = []
    theta = 2 * 3.1415926 / num_segments
    c = math.cos(theta) #precalculate the sine and cosine
    s = math.sin(theta)
    x = r # we start at angle = 0
    y = 0
    for i in range(num_segments):
        #bgl.glVertex2f(x + cx, y + cy) # output vertex
        points.append((x + cx, y + cy))
        # apply the rotation matrix
        t = x
        x = c * x - s * y
        y = s * t + c * y

    return points

def tri_circle_2d(center, verts):
    '''add center point at the beginning of the coordlist'''
    #indices = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 1)]
    return [center] + verts

def debug_display_img(img, img_name, width, height):
    blimg = bpy.data.images.get(img_name)
    if blimg:
        print('x', img.shape[0], blimg.size[1], ' y', img.shape[1], blimg.size[0])
        if img.shape[0] == blimg.size[1] and img.shape[1] == blimg.size[0]:
            pass
        else:
            bpy.data.images.remove(blimg)
            blimg = bpy.data.images.new(img_name, width=width, height=height, alpha=True, float_buffer=False)
    else:
        blimg = bpy.data.images.new(img_name, width=width, height=height, alpha=True, float_buffer=False)

    ravel_time = time()
    ravelled = (img / 255).ravel()
    print(f'ravel_time : {time() - ravel_time} secs')
    pixelimg_time = time()  
    blimg.pixels = ravelled#Loooooooong !
    print(f'pixelimg_time : {time() - pixelimg_time} secs')

### projected add stroke

def add_proj_stroke(s, frame, plane_co, plane_no, mat_id=None):
    '''Create new stroke from list'''
    obj = bpy.context.object
    mat_inv = obj.matrix_world.inverted()

    if not plane_co:#default to object location
        plane_co = obj.location

    if not plane_no:# use view depth (region_to_location instead of )
        coords_3d_flat = np.array([mat_inv @ region_to_location(co, plane_co) for co in s]).flatten()

    else:
        #projected on given plane from view (intersect on plane with a vector from view origin)
        origin = get_view_origin_position()#get view origin
        region = bpy.context.region
        rv3d = bpy.context.region_data        
        coords_3d_flat = np.array([mat_inv @ geometry.intersect_line_plane(origin,  origin - view3d_utils.region_2d_to_vector_3d(region, rv3d, co), plane_co, plane_no) for co in s]).flatten()
        #If no plane is crossed, intersect_line_plane return None which naturally goes to traceback...

    #new stroke
    ns = frame.strokes.new()#s['colorname']
    ns.display_mode = '3DSPACE'#default type is SCREEN, switch to 3DSPACE
    ns.draw_cyclic = True#virtually close the shape (as last point is not overlapping first)
    
    #use active material of active object (get material with obj.active_material)
    ns.material_index = mat_id if mat_id else obj.active_material_index
    #ns.line_width #default 0

    #add points
    pts_to_add = len(s)
    ns.points.add(pts_to_add)
    #set coordinate
    ns.points.foreach_set('co', coords_3d_flat)
    


def add_proj_multiple_strokes(stroke_list, gp=None, layer=None, use_current_frame=True, plane_co=None, plane_no=None, mat_id=None):
    '''
    add a list of strokes to active frame of given layer
    if no object is specified, active object is used
    if no layer specified, active layer is used
    if use_current_frame is True, a new frame will be created only if needed
    '''
    scene = bpy.context.scene
    #default: active gp object
    if not gp:
        if bpy.context.object.type == 'GPENCIL':
            gp = bpy.context.object.data
    gpl = gp.layers
    #default: active layer
    if not layer:
        layer = gpl.active
    fnum = scene.frame_current
    target_frame = False
    act = layer.active_frame

    for s in stroke_list:
        if act:
            if use_current_frame or act.frame_number == fnum:
                #work on current frame if exists
                # use current frame anyway if one key exist at this scene.frame
                target_frame = act

        if not target_frame:
            #no active frame
            #or active exists but not aligned scene.current with use_current_frame disabled
            target_frame = layer.frames.new(fnum)


        #Clean shape
        if isinstance(s, list):
            s = np.array(s)#already in good shape ;)
            # print(s[:2])

        if len(s.shape) == 3:#if np-array has extra dimension
            s = s[:,0] #kill the array extra dimension

        # print('\n\n\ns[0], s[-1]: ', s[0], s[-1])
        if s[0][0] == s[-1][0] and s[0][1] == s[-1][1]: #on arrays can't do if " [562. 352.] == [562. 352.]: "
            s = s[:-1]#slice off last point if same as first (no need to close)
        
        add_proj_stroke(s, target_frame, plane_co, plane_no, mat_id=mat_id)

    # print(len(stroke_list), 'strokes generated')



def gp_draw(brush, mode='NEW'):
    '''
    Get a brush as contours
    generate passed contours and hierarchy as grease pencil strokes
    in active GP object and active layer
    (in a new GP_object / layer if there isn't any)
    '''

    scn = bpy.context.scene
    #check active GP object and layer to know where to put the data
    obj = bpy.context.object
    if obj.type != 'GPENCIL':
        #create a new object or return error ? - return for now
        print('no Grease Pencil object active')
        return #silent return... else do return 'no active Grease Pencil object found'

    mat = obj.matrix_world
    gp = obj.data
    gpl = gp.layers
    layer = gpl.active
    if not layer:
        if len(gpl): layer=gpl[0]
    if not layer:#create one ?> gpl.new('GP_Layer_fill')#,set_active=True#default
        bpy.ops.gpencil.layer_add()
        layer = gpl.active
        # return 'No layer to draw in'

    frame = layer.active_frame
    if scn.GPBF_use_fill_layer:#Do things on 'fill' layers only
        if not 'fill' in layer.info.lower():
            fillname = layer.info + '_fill'
            fill_layer = gpl.get(fillname)
            if not fill_layer:
                fill_layer = gpl.new(fillname, set_active=False)#not active
                gpl.move(fill_layer, 'DOWN')#created above active so move down once
                fill_layer.frames.new(frame.frame_number, active=True)#create frame at position of existing frame on top layer
                print('Created fill layer:', fill_layer.info)
            #update layer and frame
            layer = fill_layer
            frame = layer.active_frame

    if not frame:#create frame
        frame = layer.frames.new(scn.frame_current, active=True)
    
    #material info initialisation
    mat_id = None #None set to object GP active material later

    if scn.GPBF_override_material:#property just hold the name as string
        mat_id = gp.materials.find(scn.GPBF_override_material)
        if mat_id < 0:#-1 if not found
            mat_id = None
            the_mat = bpy.data.materials.get(scn.GPBF_override_material)
            if the_mat: # auto-add material to material slot
                print(f'Automatically appended material "{scn.GPBF_override_material}" to object "{obj.name}" data')
                gp.materials.append(the_mat)
                mat_id = gp.materials.find(scn.GPBF_override_material)#refind right after append
                if mat_id < 0:
                    mat_id = None#just in case it did not work

    warn = []
    
    #SETTINGS
    settings = scn.tool_settings
    orient = settings.gpencil_sculpt.lock_axis#'VIEW', 'AXIS_Y', 'AXIS_X', 'AXIS_Z', 'CURSOR'
    loc = settings.gpencil_stroke_placement_view3d#'ORIGIN', 'CURSOR', 'SURFACE', 'STROKE'

    use_intersected_stroke_normal = False#only for add/sub -- hard coded for now

    ### CHOOSE HOW TO PROJECT

    # -> placement
    if loc == "CURSOR":
        plane_co = scn.cursor.location
    else:#ORIGIN (also on origin if set to 'SURFACE', 'STROKE')
        plane_co = obj.location
    # -> orientation
    if orient == 'VIEW':
        #only depth is important, no need to get view vector
        plane_no = None

    elif orient == 'AXIS_Y':#front (X-Z)
        plane_no = Vector((0,1,0))
        plane_no.rotate(mat)

    elif orient == 'AXIS_X':#side (Y-Z)
        plane_no = Vector((1,0,0))
        plane_no.rotate(mat)

    elif orient == 'AXIS_Z':#top (X-Y)
        plane_no = Vector((0,0,1))
        plane_no.rotate(mat)

    elif orient == 'CURSOR':
        plane_no = Vector((0,0,1))
        plane_no.rotate(scn.cursor.matrix)
    
    #debug_create_empty_from_vec(plane_co, plane_no)

    strokes = frame.strokes#if s.draw_cyclic

    ### BRUSH (should be a single nparray poly)
    ## pbrushs = [Polygon(b) for b in brushs] #cascade merge brush but maybe still multi...

    pshapes = []#out of statement so can check for additive mode later 

    if mode == 'NEW':
        new=brush#just assign brush np_array stroke

    else: #case of add/sub - handle existing strokes with shapely checks and bools
        pbrush = Polygon(brush[0][:,0])

        ## -- STROKE CHECKING AND SHAPE LISTING -- 

        ### STROKES as 2D poly --> filter on "draw_cyclic" attr (maybe add it as option)
        #get existing strokes as polygons and check if instersect

        #make a list of lists  [ [index, polygon, stroke object], ... ]
        all_pshapes = [[i, Polygon([location_to_region(mat @ p.co) for p in s.points]), s] for i, s in enumerate(strokes) if len(s.points) > 3]

        #filter intersecting shapes
        
        gpmatid = None
        gpmat = None
        if len(obj.data.materials):
            gpmatid = obj.active_material_index
            gpmat = obj.active_material

        for p in all_pshapes:
            smatid = p[2].material_index
            smat = obj.data.materials[smatid]
            if pbrush.intersects(p[1]) or pbrush.contains(p[1]) or pbrush.within(p[1]):#sub with pbrush.within(p[1]) 
                # filters
                if scn.GPBF_filter_only_fill:
                    if not smat.is_grease_pencil:continue#not gp material
                    if not smat.grease_pencil.show_fill:continue#no fill checked
                if scn.GPBF_filter_only_selected_mat:
                    if not smatid == gpmatid:continue#not same material as active one
                
                pshapes.append(p)


    if not pshapes and mode == 'SUB':
        print('brush not intersecting any strokes or stroke matching with filters')
        return #just silent return
        # return 'brush not intersecting any strokes'

    if not pshapes and mode == 'ADD':
        print('Additive mode with no overlap detected, adding a new stroke')
        mode = 'NEW'#no crossed stroke so create a new one
        new=brush#use brush

    if pshapes:
        if scn.GPBF_use_crossed_mat:
            # Get the material of the first stroke in list. a bit random but no good solution here...
            mat_id = pshapes[0][2].material_index

        if use_intersected_stroke_normal:
            #check coplanar only if projection on shape is on
            planar_shapes = []#coplanar shapes
            not_coplanar = []#index of non coplanar strokes
            for p in pshapes:
                if stroke_is_coplanar(p[2]):#retrun false if stroke have less than 4 points
                    planar_shapes.append(p)
                else:
                    not_coplanar.append(p[0])

            if not planar_shapes:
                return f'All strokes ({len(not_coplanar)}) are not coplanar'

            pshapes = planar_shapes #set pshapes with only coplanar (change that maybe later)

            if not_coplanar:
                mess = f'non coplanar strokes : {not_coplanar}'
                warn.append(mess)
                print(mess)

        ## -- BOOLEAN OPERATIONS -- 

        fused = None

        if mode == 'ADD':
            polyshapes = [p[1] for p in pshapes]
            polyshapes.append(pbrush)
            fused = cascaded_union(polyshapes)
        
        elif mode == 'SUB':
            #substract individually to shape in pshapes 
            # check multi strokes coplanarity ?
            
            # print('pshapes: ', pshapes)

            if len(pshapes) == 1:
                fused = pshapes[0][1].difference(pbrush)
                # print('fused: ', fused.geom_type)
                # print('fused.is_empty: ', fused.is_empty)
            else:
                npoly = []
                for shape in pshapes:
                    diff_result = shape[1].difference(pbrush)

                    if diff_result.geom_type == 'MultiPolygon':
                        for subpoly in diff_result:
                            # print('subpoly: ', subpoly.geom_type)
                            npoly.append(subpoly)
                    else:
                        npoly.append(diff_result)

                fused = MultiPolygon(npoly)
                # fused = MultiPolygon([shape[1].difference(pbrush) for shape in pshapes])

        """ if not fused:
            mess = 'problem as occured with boolean operation'
            print(mess)
            return mess """

        new = []
        if fused.geom_type == 'Polygon':
            # print('is simple poly')
            new.append(np.array(fused.exterior.coords))
        
        else:#'MultiPolygon'
            #print('is multi poly :'.upper(), len(fused) )
            for poly in fused:
                new.append(np.array(poly.exterior.coords))

        if use_intersected_stroke_normal:#override planar projection settings
            ### get intersected stroke point and normal
            a,b,c = pshapes[0][2].points[0].co, pshapes[0][2].points[1].co, pshapes[0][2].points[-2].co
            plane_co = a
            plane_no = get_normal_from_tri(a,b,c)
            print('cross_vec: ', cross_vec)
            
            if cross_vec == Vector((0, 0, 0)):
                mess = 'Problem, Normal of first intersect stroke is evaluated as Vector(0,0,0)'
                print(mess)
                return mess


        # Check Angle from view
        view_direction = view3d_utils.region_2d_to_vector_3d(bpy.context.region, bpy.context.region_data, (bpy.context.region.width/2.0, bpy.context.region.height/2.0))
        # print('plane_no: ', plane_no)
        angle = math.degrees(view_direction.angle(plane_no))
        # correct angle value when painting from other side (seems a bit off...)
        if angle > 90: angle = abs(90 - (angle - 90))

        #over-angle error
        if angle > 75:
            return f"painting on a surface with angle over: 75 degrees ({angle:.2f})"

        #over angle warning 
        elif angle > 45:
            mess = f"painting on a surface with angle over: 45 degrees ({angle:.2f})"
            # print(mess)
            warn.append(mess)
        
        if angle > 5: print('angle from view: {:.2f}'.format(angle)) 

    if new:
        add_proj_multiple_strokes(new, layer=layer, plane_co=plane_co, plane_no=plane_no, mat_id=mat_id)

    need_update = False
    if mode != 'NEW':# delete original strokes in pshape
        if pshapes:
            need_update = True
        #optional: attr transfer/copy (attribute replication for points in same place / aligned)
        for p in sorted(pshapes, reverse=True):
            if need_update:
                # only deleting not update in viewport if there is only suppression...
                # so deleting point first to get a visual update
                for i in range(len(p[2].points)):
                    p[2].points.pop()

            strokes.remove(p[2])


    if warn:
        return warn



class GPBF_OT_open_doc(bpy.types.Operator):
    bl_idname = "gp.bf_open_doc"
    bl_label = "Open webpage documentation"
    bl_description = "Open doc for module installation"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        import webbrowser
        url = ""
        webbrowser.open(url)
        return {"FINISHED"}

def missing_module_popup_panel(self, context):
    layout = self.layout
    layout.label("This functionality use opencv and shapely modules")
    # layout.label("(manual installation notes are diplayed in console)")
    layout.label("Check how to install modules in the readme")
    # layout.operator('materials.dl_webcolors_module')
    layout.operator('gp.bf_open_doc')


def draw_callback_px(self, context):
    scn = context.scene
    # 50% alpha, 2 pixel width line
    shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
    bgl.glEnable(bgl.GL_BLEND)
    bgl.glLineWidth(2)

    # paint
    batch = batch_for_shader(shader, 'TRIS', {"pos": self.vertices}, indices=self.indices)
    shader.bind()
    shader.uniform_float("color", self.paint_color)#indigo (0, 0.5, 0.5, 1.0)
    batch.draw(shader)

    '''
    #draw debug line showing mouse path
    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": self.mouse_path})
    shader.bind()
    shader.uniform_float("color", (0.5, 0.5, 0.5, 0.5))#grey-light
    batch.draw(shader)
    '''

    #paint widget
    #paint_widget = Point(self.mouse).buffer(scn.GPBF_radius, 2)#shapely mode !
    
    paint_widget = circle_2d(self.mouse, self.pen_radius, self.crosshair_resolution)#optimisation ?
    paint_widget.append(paint_widget[0])#re-insert last coord to close the circle

    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": paint_widget})
    shader.bind()
    shader.uniform_float("color", (0.6, 0.0, 0.0, 0.5))#red-light
    batch.draw(shader)

    # restore opengl defaults
    bgl.glLineWidth(1)
    bgl.glDisable(bgl.GL_BLEND)

    ## text

    font_id = 0
    ## show active modifier
    if self.pressed_alt or self.pressed_shift:
        blf.position(font_id, self.mouse[0]+scn.GPBF_radius, self.mouse[1]+scn.GPBF_radius, 0)
        blf.size(font_id, 15, 72)
        if self.pressed_alt:
            blf.draw(font_id, '-')
        else:
            blf.draw(font_id, '+')

    ## draw text debug infos
    blf.position(font_id, 15, 30, 0)
    blf.size(font_id, 20, 72)
    blf.draw(font_id, f'Flat paint -  radius: {scn.GPBF_radius} spacing: {scn.GPBF_spacing}')
    # blf.draw(font_id, f'radius: {scn.GPBF_radius} spacing: {scn.GPBF_spacing} points: {self.points_num} mouse_steps: {len(self.mouse_path)} lenght: {self.distance}')


class GP_OT_draw_fill(bpy.types.Operator):
    """Draw with the mouse"""
    bl_idname = "view3d.gp_fill_brush_draw"
    bl_label = "Brush fill"
    bl_description = "Draw/add/erase filled stroke.\nShortcuts: ctrl+shift+F\nUse shift to add, alt to erase\nChange radius with mousewheel or []"
    bl_options = {"REGISTER", "UNDO"}

    pressed_key = 'NOTHING'
    pressed_alt = False
    pressed_ctrl = False
    pressed_shift = False
    points_num = 0#the number of point drawn
    pen_radius = 10#radius calculated from base

    def reset_stroke(self):
        self.mouse_path = [] #coordinate list of the mouse path
        self.indices = [] #all indices of stroke
        self.distance = 0.0 #just initialise lenght of
        self.all_points = []
        self.all_radius = []
        self.polygons = []
        self.pressed_key = 'NOTHING'

    def modifier_key_state(self, modkeys, event):
        if event.type in modkeys:
            if event.value == 'PRESS':
                return True
        return False#dont need :elif event.value == 'RELEASE':return False

    def modal(self, context, event):
        context.area.tag_redraw()
        scn = context.scene
        #handle modifier keys:
        if event.type in {'LEFT_SHIFT', 'RIGHT_SHIFT', 'LEFT_ALT', 'RIGHT_ALT', 'LEFT_CTRL', 'RIGHT_CTRL'}:
            self.pressed_shift = self.modifier_key_state({'LEFT_SHIFT', 'RIGHT_SHIFT'}, event)
            self.pressed_alt = self.modifier_key_state({'LEFT_ALT', 'RIGHT_ALT'}, event)
            # self.pressed_ctrl = self.modifier_key_state({'LEFT_CTRL', 'RIGHT_CTRL'}, event)

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:#mouse sub-moves too to get higher resolution of mouse moves
            self.mouse_path.append((event.mouse_region_x, event.mouse_region_y))
            self.mouse = (event.mouse_region_x, event.mouse_region_y)

        self.pen_radius = scn.GPBF_radius
        # self.reticule = circle_2d(self.mouse, scn.GPBF_radius, self.crosshair_resolution)

        #handle the continuous press
        if event.type == 'LEFTMOUSE' :
            self.pressed_key = 'LEFTMOUSE'
            #while pushed, variable pressed stay on...
            if event.value == 'RELEASE':
                #if release, stop and do the thing !
                
                ### UNDO STEP push before creating new stroke
                bpy.ops.ed.undo_push()
                
                self.pen_radius = scn.GPBF_radius#show max radius when not drawing

                if not self.all_points:
                    #If no mouse-move in list (hasn't move), just do nothing and reset brush
                    self.reset_stroke()
                    return {'RUNNING_MODAL'}

                ### "paint" all circles on a numpy array of the size of the window
                start = time()

                width, height = context.area.width, context.area.height#initiate numpy array by area size
                #maybe multiply size by 2/3/4 for more resolution when painting circles and downscale after findContours (in the comprehension list before approxPolyDP)
                rfactor = scn.GPBF_resolution_factor
                img = np.zeros((height*rfactor, width*rfactor, 4), dtype=np.uint8)

                ### fill it with openCV
                for p, r in zip(self.all_points, self.all_radius):
                    cv2.circle(img, (p[0]*rfactor, p[1]*rfactor), r*rfactor, (255,255,255,255), thickness=-1, lineType=cv2.LINE_AA)
                    # cv2.circle(img, p, r, (255,255,255,255), thickness=-1, lineType=cv2.LINE_AA)

                ## blur to slightly smooth shape (but extend stroke borders)
                #img = cv2.blur(img, (2,2))
                # debug_display_img(img, 'pixel', width, height)

                ## get contour
                ## ref : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_hierarchy/py_contours_hierarchy.html
                imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                contours, hierarchy = cv2.findContours(imgray,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #RETR_EXTERNAL (only external) #RETR_CCOMP dual hierarchy...

                ## approximation
                ## good value : 0.002 between 0.0011 and 0.002 but angle visible before
                # simplify = 0.0015
                # simplify = scn.GPBF_brush_approx#0.0015

                ## for a minimum of 0.0003, 14 is a good default value. for 0.0008, 8 is good
                # simplify = round( transfer_value(scn.GPBF_brush_approx, 0, 100, 0.0003, 0.01) , 5)
                simplify = scn.GPBF_brush_approx / 10000

                # print('simplify: ', scn.GPBF_brush_approx, '>', simplify)

                #ref : https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-approximation
                contours = [cv2.approxPolyDP(cnt, simplify*cv2.arcLength(cnt,True), True) for cnt in contours]
                # contours = [cv2.approxPolyDP(np.array([c/rfactor for c cnt]), simplify*cv2.arcLength(cnt,True), True) for cnt in contours]

                if not contours:
                    print('No contour found on brush, exit modal')
                    return {'CANCELLED'}
                    #self.report({'ERROR'}, 'no contour in brush list')
                    #return {'RUNNING_MODAL'}
                    
                # print(type(contours[0]))
                # print(contours[0][:2])
                contours = [i/rfactor for i in contours]
                # print(contours[0][:2])

                # contours = [cv2.approxPolyDP(cnt / rfactor, simplify*cv2.arcLength(cnt / rfactor,True), True) for cnt in contours]

                '''#debug draw contour in bl_image
                contour_img = np.zeros((height, width, 4), dtype=np.uint8)
                contour_img = cv2.drawContours(contour_img, contours[0], -1, (255,0,0,1), 3)
                contour_img = np.array(contour_img * 255, dtype = np.uint8)
                debug_display_img(contour_img, 'contour', width, height)
                '''

                ### Deal with generated brush shape and send to draw func
                    
                if self.pressed_alt:#eraser operation
                    err = gp_draw(contours, mode='SUB')
                
                elif self.pressed_shift:#additive union operation
                    err = gp_draw(contours, mode='ADD')
                
                else:#plain stroke
                    err = gp_draw(contours, mode='NEW')

                # print(f'stroke creation time : {time() - start} secs')
                ## reset all properties for next stroke
                self.reset_stroke()

                #Show error
                if err:
                    if isinstance(err,list):#warnings are list
                        self.report({'WARNING'}, '\n'.join(err))
                    else:#error are str
                        self.report({'ERROR'}, err)


        if self.pressed_key == 'LEFTMOUSE':

            #set pressured radius (disable in a condition to get always full force)
            self.pen_radius = int(scn.GPBF_radius * event.pressure)
            #debug pressure
            # print('pression', event.pressure, '-> self.pen_radius: ', self.pen_radius)

            if self.pen_radius < 1:
                self.pen_radius = 1

            if len(self.mouse_path) > 1:#need two element to calculate lenght

                last, prev = self.mouse_path[-1], self.mouse_path[-2]
                new_dist = vector_length_2d(prev, last)#get lenghts of last stroke
                self.distance += new_dist#update full lenght

                prev_pt_num = self.points_num
                self.points_num = int( self.distance // scn.GPBF_spacing ) #euclidian div (back to int)
                # all points
                new_points = self.points_num - prev_pt_num#
                # print('new_points', new_points)#Dbg

                vertices = []
                for i in range(new_points):
                    ##  find position on the last segment >> how to place on "Full" line if sapcing is larger than last segment :s
                    #!! approximation using only last segment
                    ##  only add if last point in mouse path is >= in distance of current point.

                    ### Calculate distance between self.mouse_path[] only with the two last point)
                    #print('ms:',len(self.mouse_path))

                    p = point_from_dist_in_segment_2d(prev, last, scn.GPBF_spacing * i+1, new_dist)
                    self.all_points.append(p)

                    # self.all_radius.append()
                    self.all_radius.append(self.pen_radius)#with pressure else: scn.GPBF_radius

                    circle = circle_2d(p, self.pen_radius, self.crosshair_resolution)
                    vertices = tri_circle_2d(p, circle)
                    #add previous number to indices in each tuple of the list
                    indices = [tuple([i + len(self.vertices) for i in tup]) for tup in self.crosshair_indices]#id_array = np.array(indices]) + len(self.vertices)

                    #add new verts/id to the list
                    self.vertices += vertices
                    self.indices += indices
            else:
                #draw at cursor location if no calculation are possible

                #sply_point = Point(self.mouse).buffer(self.pen_radius)
                self.all_points.append(self.mouse)
                self.all_radius.append(self.pen_radius)

                circle = circle_2d(self.mouse, self.pen_radius, self.crosshair_resolution)
                vertices = tri_circle_2d(self.mouse, circle)
                indices = [tuple([i + len(self.vertices) for i in tup]) for tup in self.crosshair_indices]
                
                #add new verts/id to the list
                self.vertices += vertices
                self.indices += indices

            #bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            #return {'FINISHED'}

        #change brush size
        if event.type in {'NUMPAD_MINUS', 'LEFT_BRACKET', 'W', 'WHEELDOWNMOUSE'}:
            if event.value == 'PRESS':
                scn.GPBF_radius -= 1
                if scn.GPBF_radius < 1:
                    #force minimal radius value to 1px... might be better way to do this
                    scn.GPBF_radius = 1

        if event.type in {'NUMPAD_PLUS', 'RIGHT_BRACKET', 'E', 'WHEELUPMOUSE'}:
            if event.value == 'PRESS':
                scn.GPBF_radius += 1

        #change spacing
        if event.type in {'S'}:
            if event.value == 'PRESS':
                scn.GPBF_spacing -= 1

        if event.type in {'D'}:
            if event.value == 'PRESS':
                scn.GPBF_spacing += 1


        #abort
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            #timer time
            # context.window_manager.event_timer_remove(self.draw_event)
            # print('STOPPED')
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')


            return {'CANCELLED'}

        """### keycode printer
        if event.type not in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            print('key:', event.type, 'value:', event.value)
        """

        return {'RUNNING_MODAL'}


    def invoke(self, context, event):
        # print('\nSTARTED')

        """#problem, import are not transfered to modal. just rely on classic import...
        # here check if opencv and shapely are Ok
        module_missing = []
        import importlib
        if not importlib.util.find_spec("cv2"): module_missing.append('cv2')
        if not importlib.util.find_spec("shapely"): module_missing.append('shapely')
        if module_missing:
            mess = 'Missing modules ' + ' and '.join(module_missing)
            self.report({'ERROR'}, mess)#WARNING, INFO
            bpy.context.window_manager.popup_menu(missing_module_popup_panel, title="Modules", icon='INFO')
            return {'CANCELLED'}

        #massive import
        import cv2#openCV
        import shapely### shapely
        from shapely.geometry import LineString, MultiPoint, Point, Polygon, MultiPolygon
        from shapely.ops import split, cascaded_union
        """
        if not context.area.spaces[0].region_3d.is_perspective:
            self.report({'ERROR'}, "Impossible to paint in orthographic view")
            return {'CANCELLED'}

        if context.area.type == 'VIEW_3D':
            args = (self, context)
            # Add the region OpenGL drawing callback
            # draw in view space with 'POST_VIEW' and 'PRE_VIEW'
            self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
            #timer time
            # self.draw_event = context.window_manager.event_timer_add(0.1, window=context.window)
            
            # paint settings
            prefs = get_addon_prefs()
            self.paint_color = prefs.GPBF_paint_color#(0.0, 0.5, 0.5, 1.0)
            if prefs.GPBF_use_material_color:
                if context.object.active_material.is_grease_pencil:
                    if context.object.active_material.grease_pencil.show_fill:
                        self.paint_color = tuple([i+0.15 for i in context.object.active_material.grease_pencil.fill_color[:3]] + [0.8])

            self.crosshair_color = prefs.GPBF_cursor_color#(0.6, 0.0, 0.0, 0.5)
            ## crosshair resolution #should be adaptative according to radius and spacing...
            # self.crosshair_resolution = 12#hardcode for now(4,8,12,16...) keep multiple of 4
            self.crosshair_resolution = prefs.GPBF_brush_display_res * 4

            # self.shapely_buffer_res = int(self.crosshair_resolution / 4)# convert to shapely buffer res (-> number * 4)

            #self.crosshair_indices = [(0,i+1, i+2) for i in range(resolution*4-1)]+[(0,resolution*4, 1)]
            self.crosshair_indices = [(0,i+1, i+2) for i in range(self.crosshair_resolution-1)] + [(0,self.crosshair_resolution, 1)]
            #print(self.crosshair_indices)
            #print('[(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 1)]')


            self.mouse_path = [] #coordinate list of the mouse path
            self.mouse = (0, 0) #actualised tuple of mouse coordinate
            # self.reticule = [] #list ot point of the cicrle widget
            self.indices = [] #all indices of stroke
            self.vertices = [] #all vertices of stroke
            self.distance = 0.0 #just initialise lenght of stroke
            self.all_points = []
            self.all_radius = []
            
            ### now scene properties :
            ## self.radius = 10 #size of the brush 
            ## self.spacing = 2 #define a spacing in px

            #polygons handling
            self.polygons = []

            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}


class GPBF_addon_prefs(bpy.types.AddonPreferences):
    bl_idname = __name__
    #some_bool_prop to display in the addon pref

    GPBF_paint_color : bpy.props.FloatVectorProperty(name="Brush paint color", 
    description="Change the 'painting' color of the brush", 
    default=(0.0, 0.5, 0.5, 1.0), step=3, precision=2, subtype='COLOR', size=4)
    
    GPBF_cursor_color : bpy.props.FloatVectorProperty(name="Brush cursor color", 
    description="Change the circle brush color", 
    default=(0.6, 0.0, 0.0, 0.5), step=3, precision=2, subtype='COLOR', size=4)

    #display/cosmetic/HUD/OSD settings
    GPBF_brush_display_res : bpy.props.IntProperty(name="Brush cursor display smoothness", 
    description="Resolution of the brush circle and paint display.\nJust convenience VS performance. Does not affect final stroke aspect, higher value might slow down painting",
    default=3, min=1, max=8, soft_min=2, soft_max=6, step=1)#, options={'HIDDEN'}#subtype = 'PIXEL' ?

    GPBF_use_material_color : bpy.props.BoolProperty(name="Display material color", 
    description="If available, display material fill color as temporary paint color", default=True)

    GPBF_register_shortcut_default : bpy.props.BoolProperty(name="register default shortcut (Ctrl+shift+F)", 
    description="Use F like Fill. If changed, need to restart blender to take effect\nregister this keymap", 
    default=False)

    def draw(self, context):
        layout = self.layout
        layout.label(text='Cosmetic changes :')
        box = layout.box()
        box.label(text="Note : These preferences does not affect aspect of final stroke,")
        box.label(text="even if the temporary displayed brush and paint apperas different")
        layout.prop(self, "GPBF_brush_display_res")
        row = layout.row(align=True)
        row.prop(self, "GPBF_cursor_color")
        row.prop(self, "GPBF_paint_color")
        layout.prop(self, "GPBF_use_material_color")

        ## keymap handling
        layout.separator()
        layout.label(text="Keymap management :")
        layout.prop(self, "GPBF_register_shortcut_default")
        layout.label(text="To create your own shortcut:")
        layout.label(text="Make sure above shortcut is disabled (if not, disable > save preferences > restart blender)")
        layout.label(text="Then right click on brush fill button in interface and add your own shortcut.")
        layout.label(text="If you prefer to manually add you keymap, the identifier of the operator is 'view3d.gp_fill_brush_draw'")
        


##  Base panel
class GP_PT_brush_fill_panel(bpy.types.Panel):
    # bl_idname = "GP_PT_brush_fill_panel"# identifier, if ommited, takes the name of the class.
    bl_label = "Brush fill"# title
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"#toolbar -  "TOOLS" for sidebar
    bl_category = "Gpencil"#name of the tab

    @classmethod
    def poll(cls, context):
        return (context.object is not None and context.object.type == 'GPENCIL')

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        layout.operator("view3d.gp_fill_brush_draw", icon='BRUSH_DATA')

        layout.prop(context.scene, 'GPBF_radius')
        layout.prop(context.scene, 'GPBF_spacing')

        layout.prop_search(context.scene, "GPBF_override_material",  bpy.data, "materials")#material selector for fill only...

        # fill layer painting option
        layout.prop(context.scene, 'GPBF_use_fill_layer')
        if context.scene.GPBF_use_fill_layer:
            if 'fill' in context.object.data.layers.active.info.lower():
                layout.label(text=f"draw on: {context.object.data.layers.active.info}")
            else:
                layout.label(text=f"draw on: {context.object.data.layers.active.info + '_fill'}")

## Sub panels
class GP_PT_brush_fill_quality_subpanel(bpy.types.Panel):
    bl_label = "Quality options"# title
    bl_parent_id = "GP_PT_brush_fill_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'#'TOOLS'
    bl_category = "Gpencil"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(context.scene, 'GPBF_resolution_factor')
        layout.prop(context.scene, 'GPBF_brush_approx')


class GP_PT_brush_fill_filter_subpanel(bpy.types.Panel):
    bl_label = "Filter Add/Sub strokes"#"Strokes filters"
    bl_parent_id = "GP_PT_brush_fill_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'#'TOOLS'
    bl_category = "Gpencil"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.label(text="Those settings only affect additive or eraser mode")
        layout.prop(context.scene, 'GPBF_use_crossed_mat')
        layout.prop(context.scene, 'GPBF_filter_only_fill')
        layout.prop(context.scene, 'GPBF_filter_only_selected_mat')


# -----------------
### Register
# -----------------

classes = (
GP_OT_draw_fill,
GP_PT_brush_fill_panel,
GP_PT_brush_fill_quality_subpanel,
GP_PT_brush_fill_filter_subpanel,
GPBF_OT_open_doc,
GPBF_addon_prefs
)

addon_keymaps = []

def register():
    ## properties basic settings
    bpy.types.Scene.GPBF_radius = bpy.props.IntProperty(name="Radius", 
    description="Radius of the fill brush\nUse [/], W/E, numpad -/+ or mousewheel down/up to modify during draw", 
    default=12, min=1, max=500, soft_min=0, soft_max=300, step=1)#, options={'HIDDEN'}#subtype = 'PIXEL' ?

    bpy.types.Scene.GPBF_spacing = bpy.props.IntProperty(name="Spacing", 
    description="Spacing of the fill brush along draw movement\nUse S/D to modify during draw", 
    default=2, min=0, max=100, soft_min=1, soft_max=50, step=1)#, options={'HIDDEN'}#subtype = 'PIXEL' ?

    ## paint option
    bpy.types.Scene.GPBF_use_fill_layer = bpy.props.BoolProperty(name="Paint on fill layer", 
    description="If not 'fill' in layer name, add new strokes to a layer 'active_layer_name' + '_fill', create if necessary", 
    default=False)

    ## quality settings
    bpy.types.Scene.GPBF_resolution_factor = bpy.props.IntProperty(name="Definition multiply", 
    description="More precise painting, add definition at the cost of speed (at the moment of releasing pencil).\nVirtually upscale pixel resolution by given factor (before converting pixels to polygons)\n/!\ You should lower approximation as you increase this value", 
    default=4, min=1, max=20, soft_min=1, soft_max=12, step=1)

    bpy.types.Scene.GPBF_brush_approx = bpy.props.FloatProperty(name="Shape approx",
    description="Approximation of the painted shape (when converting from pixel to GP polygon).\nHigh value means more approximate, value too low make the pixel visible, usually keep between\nHints: you can enter value above 100 for greater approximation",
    default=10, min=0, max=1000, soft_min=0, soft_max=100, step=1, precision=1, subtype='NONE', unit='NONE')
    # realvalues : default=0.0015, min=0.0, max=0.1, soft_min=0.0014, soft_max=0.05, step=0.01, precision=4

    ## filter settings
    bpy.types.Scene.GPBF_use_crossed_mat = bpy.props.BoolProperty(name="Keep existing material", 
    description="In additive or eraser mode, take the material of crossed object when recreating the line\n/!\ If multiple material crossed this will be random !",
    default=False)
    
    bpy.types.Scene.GPBF_filter_only_fill = bpy.props.BoolProperty(name="Only affect fill ", 
    description="Affect only strokes that have a fill active in used material", 
    default=True)#buggy hen interacting with normal strokes
    
    bpy.types.Scene.GPBF_filter_only_selected_mat = bpy.props.BoolProperty(name="Only selected mat", 
    description="Affect only strokes that use current selected material", 
    default=False)

    ##material selector
    bpy.types.Scene.GPBF_override_material = bpy.props.StringProperty(name="Fill material",
    description="If specified, use this material only for the brush fill\nelse use active material\n(limitation : does not follow material name change)")

    ## class
    for cls in classes:
        bpy.utils.register_class(cls)

    ## keymap
    if get_addon_prefs().GPBF_register_shortcut_default:
        kcfg = bpy.context.window_manager.keyconfigs.addon
        if kcfg:
            km = kcfg.keymaps.new(name='3D View', space_type='VIEW_3D')

            kmi = km.keymap_items.new("view3d.gp_fill_brush_draw", 'F', 'PRESS', shift=True, ctrl=True)

            addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.GPBF_radius
    del bpy.types.Scene.GPBF_spacing
    del bpy.types.Scene.GPBF_brush_approx
    del bpy.types.Scene.GPBF_use_crossed_mat
    del bpy.types.Scene.GPBF_filter_only_fill
    del bpy.types.Scene.GPBF_filter_only_selected_mat
    del bpy.types.Scene.GPBF_resolution_factor
    del bpy.types.Scene.GPBF_use_fill_layer
    del bpy.types.Scene.GPBF_override_material


if __name__ == "__main__":
    register()