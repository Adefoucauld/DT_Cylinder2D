DefineConstant[
jets_toggle = {1, Name "Toggle Jets --> 0 : No jets, 1: Yes jets"}
height_cylinder = {1, Name "Cylinder Height (ND)"}
ar = {1.0, Name "Cylinder Aspect Ratio"}
cylinder_y_shift = {0.0, Name "Cylinder Center Shift from Centerline, Positive UP (ND)"}
x_upstream = {10, Name "Domain Upstream Length (from left-most rect point) (ND)"}
x_downstream = {26, Name "Domain Downstream Length (from right-most rect point) (ND)"}
height_domain = {20, Name "Domain Height (ND)"}
coarse_y_distance_top_bot = {4, Name "y-distance from center where mesh coarsening starts"}
coarse_x_distance_left_from_LE = {2, Name "x-distance from upstream face where mesh coarsening starts"}
mesh_size_cylinder = {0.05, Name "Mesh Size on Cylinder Walls"}
mesh_size_jets = {0.01, Name "Mesh Size on jet suirfaces"}
mesh_size_medium = {0.2, Name "Medium mesh size (at boundary where coarsening starts"}
mesh_size_coarse = {1, Name "Coarse mesh Size Close to Domain boundaries outside wake"}
jet_width = {0.1, Name "Jet Width (ND)"}
];

// Seed the cylinder's center's identifier and create the center point
center = newp;
Point(center) = {0, 0, 0, mesh_size_cylinder};

// Cylinder dimensions
r_height = height_cylinder; // Cylinder height
r_length = ar*height_cylinder; // Cylinder length


// Start definition of cylinder surfaces (curves). Note: it is defined in CCW sense
// Each jet surface is a physical line, and the remaining of the rectangle is another

// Define x,y coors of the rectangle sides
y_top = r_height/2;
y_bot = -r_height/2;
x_left = -r_length/2;
x_right = r_length/2;

// Define x coors of jets centres and upstream bound
x_jet_centre = x_right-jet_width/2;
x_jet_leftbound = x_right-jet_width;

//Define y coors of base jets centres and bounds
y_jet_top_centre = y_top-jet_width/2;
y_jet_bot_centre = y_bot+jet_width/2;
y_jet_topbound = y_top-jet_width;
y_jet_botbound = y_bot+jet_width;


// Define all points of rectangle
p = newp;
Point(p) = {x_right, y_top, 0, mesh_size_jets};  // Top right corner (p)
Point(p+1) = {x_jet_centre, y_top, 0, mesh_size_jets};  // Top jet centre (p+1)
Point(p+2) = {x_jet_leftbound, y_top, 0, mesh_size_jets};  // Top jet upstream bound (p+2)
Point(p+3) = {x_left, y_top, 0, mesh_size_cylinder};  // Top left corner (p+3)
Point(p+4) = {x_left, y_bot, 0, mesh_size_cylinder};  // Bottom left corner (p+4)
Point(p+5) = {x_jet_leftbound, y_bot, 0, mesh_size_jets};  // Bottom jet upstream bound (p+5)
Point(p+6) = {x_jet_centre, y_bot, 0, mesh_size_jets};  // Bottom jet centre (p+6)
Point(p+7) = {x_right, y_bot, 0, mesh_size_jets};  // Bottom right corner (p+7)
Point(p+8) = {x_right, y_jet_bot_centre, 0, mesh_size_jets};  // Base jet bottom centre (p+8)
Point(p+9) = {x_right, y_jet_botbound, 0, mesh_size_jets};  // Base jet bottom bound (p+9)
Point(p+10) = {x_right, y_jet_topbound, 0, mesh_size_jets}; // Base jet top bound (p+10)
Point(p+11) = {x_right, y_jet_top_centre, 0, mesh_size_jets}; // Base jet top centre (p+11)

Point(p+12) = {x_right, y_jet_topbound - r_height/20, 0, mesh_size_cylinder};  // Auxiliary point for base top 
Point(p+13) = {x_right, y_jet_botbound + r_height/20, 0, mesh_size_cylinder};  // Auxiliary point for base bottom

If(jets_toggle)

  cylinder[] = {}; // Create empty list of curves (surfaces) of the cylinder. Defined CCW
  no_slip_cyl[] = {};  // No-slip cylinder physical surfaces list
  
  //Define top jet surface:
  l = newl;
  Line(l) = {p,p+1};
  Line(l+1) = {p+1, p+2};
  Physical Line(5) = {l,l+1}; //Top jet physical surface
  cylinder[] += {l,l+1}; //Add to cylinder list

  // Define left no-slip surface of cylinder
  l = newl;
  Line(l) = {p+2, p+3};
  Line(l+1) = {p+3, p+4};
  Line(l+2) = {p+4, p+5};
  no_slip_cyl[] += {l, l+1, l+2};
  cylinder[] += {l, l+1, l+2};

  // Define bottom jet surface:
  l = newl;
  Line(l) = {p+5, p+6};
  Line(l+1) = {p+6, p+7};
  Physical Line(6) = {l, l+1};  // Bottom jet physical surface
  cylinder[] += {l, l+1}; // Add to cylinder list

  //Define base bottom jet surface:
  l = newl;
  Line(l) = {p+7, p+8};
  Line(l+1) = {p+8, p+9};
  Physical Line(7) = {l, l+1};  // Bottom jet physical surface
  cylinder[] += {l, l+1}; // Add to cylinder list

  // Define right no-slip surface
  l = newl;
  Line(l) = {p+9, p+13};
  Line(l+1) = {p+13, p+12};
  Line(l+2) = {p+12, p+10};
  no_slip_cyl[] += {l, l+1, l+2};
  cylinder[] += {l, l+1, l+2}; // Add to cylinder list

  Physical Line(4) = {no_slip_cyl[]};  // Define no-slip cylinder physical surfaces
  
  // Define base top jet surface:
  l = newl;
  Line(l) = {p+10, p+11};
  Line(l+1) = {p+11, p};
  Physical Line(8) = {l, l+1};  // Bottom jet physical surface
  cylinder[] += {l, l+1}; // Add to cylinder list


// Just the rectangle (if number no jets)
Else

   l = newl;
   Line(l) = {p, p+3};
   Line(l+1) = {p+3, p+4};
   Line(l+2) = {p+4, p+7};
   Line(l+3) = {p+7, p};

   cylinder[] = {l, l+1, l+2, l+3};	// List of curves (surfaces) of the cylinder. Defined CCW
   Physical Line(4) = {cylinder[]}; // Define no-slip cylinder physical surfaces (in this case all cyl)
EndIf

// Create the channel (Domain exterior boundary)
// Define useful quantities
y_top_dom = height_domain/2-cylinder_y_shift;  // Smaller than half the height if positive shift
y_bot_dom = -height_domain/2-cylinder_y_shift; // Larger in mag than half the height if positive shift
x_left_dom = -r_length/2-x_upstream;
x_right_dom = r_length/2+x_downstream;

y_coarse_top = coarse_y_distance_top_bot;
y_coarse_bot = - coarse_y_distance_top_bot;
x_coarse_left = - r_length/2 - coarse_x_distance_left_from_LE;

// Define points
p = newp;
Point(p) = {x_left_dom, y_bot_dom, 0, mesh_size_coarse}; // domain bottom-left corner
Point(p+1) = {x_right_dom, y_bot_dom, 0, mesh_size_coarse}; // domain bottom-right corner
Point(p+2) = {x_right_dom, y_top_dom, 0, mesh_size_coarse}; // domain top-right corner
Point(p+3) = {x_left_dom, y_top_dom, 0, mesh_size_coarse}; // domain top-left corner

Point(p+4) = {x_coarse_left, y_coarse_bot, 0, mesh_size_medium}; // coarsening bottom-left corner
Point(p+5) = {x_right_dom, y_coarse_bot, 0, mesh_size_medium}; // coarsening bottom-right corner
Point(p+6) = {x_right_dom, y_coarse_top, 0, mesh_size_medium}; // coarsening top-right corner
Point(p+7) = {x_coarse_left, y_coarse_top, 0, mesh_size_medium}; // coarsening top-left corner


l = newl;
// Bottom wall (slip-free)
Line(l) = {p, p+1};
Physical Line(1) = {l};

// Right wall (outflow)
Line(l+1) = {p+1, p+5};  // Bottom-right side
Line(l+2) = {p+5, p+6};  // Middle-right side (coarsening bound right)
Line(l+3) = {p+6, p+2};  // Top-right side
Physical Line(2) = {l+1, l+2, l+3};

// Top wall (slip free)
Line(l+4) = {p+2, p+3};
Physical Line(1) += {l+4};

// Inlet
Line(l+5) = {p+3, p};
Physical Line(3) = {l+5};

// Coarsening bound bottom
Line(l+6) = {p+4, p+5};

// Coarsening bound top
Line(l+7) = {p+6, p+7};

// Coarsening bound left
Line(l+8) = {p+7, p+4};

// Define coarse mesh portion of domain
// Create line loop for coarse area
coarse = newll;
Line Loop(coarse) = {(l), (l+1), -(l+6), -(l+8), -(l+7), (l+3), (l+4), (l+5)};
// Create surface and physical surface for coarse area
s = news;
Plane Surface(s) = {coarse};
Physical Surface(1) = {s};  // Physical surface to be mesh (then we'll add fine portion)

// Create line loop for fine area (containing the cylinder)
fine_outer = newll;
Line Loop(fine_outer) = {(l+6), (l+2), (l+7), (l+8)};  // Outer line loop of fine zone
fine_inner = newll;
Line Loop(fine_inner) = {cylinder[]}; // Inner line loop (cylinder)

// Define final physical surface
s = news;
Plane Surface(s) = {fine_outer, fine_inner}; // Should be outer, inner, no??
Physical Surface(1) += {s}; // // Add to surface to be mesh


// First the jet and no slip surfaces of the cylinder are defined. Each jet surface is a physical line and all the no slip
// cylinder surfaces are another. Then the domain is created.
