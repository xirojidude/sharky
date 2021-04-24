
Shader "Skybox/EnglishLane"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #define mod(x,y) (x-y*floor(x/y)) // glsl mod
            #define iTime _Time.y
            #define iResolution _ScreenParams
            #define vec2 float2
            #define vec3 float3
            #define vec4 float4
            #define mix lerp
            #define texture tex2D
            #define fract frac
            #define mat4 float4x4
            #define mat3 float3x3
            #define mat2 float2x2
            #define textureLod(a,b,c) tex2Dlod(a,float4(b,0,c))
            #define atan(a,b) atan2(b,a)

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }


// English Lane by Jerome Liard, April 2021
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// https://www.shadertoy.com/view/fsXXzX
//
// You are walking and flying through an infinite English countryside.
// Chill out and use the mouse to look around. 
// A single walk->fly cycle is about 50s.
//
// Shadertoy compilation time seems to be about 15s, thanks for your patience.

// first lane index, at each walk-flight cycle we switch to next lane midair
#define FIRST_LANE_INDEX 10.0

// If the reprojection is janky please press the button that resets time to zero.
//
// I wanted to make a navigable countryside with paths inspired by paintings from Richard Thorn (see his book "Down an English Lane"), 
// and a little bit by Hiroshi Nagai and Tezuka Osamu's Jumping short anime (both life long inspirations).
//
// Creation of the initial patchwork and parameterized paths network:
//
//   - 2 perpendicular sets of regularly spaced parallel 1d lanes are used. 
//   - Each 1d lane has an id. The amplitude of each 1d lane must be such that they don't cross the previous or next 1d lane.
//   - The horizonal set of parallel lanes have constant vertical center spacing.
//   - The twist: the vertical set of parallel lanes can have their spacing set more freely based on which stab we are in the horizontal set.
//     This helps generating complex branching patterns.
//   - For each set of parallel lanes we simply use its local x coordinate as a parameter (used for garden brick wall and camera).
//   - The intersections of lane stabs give us a cellular base for country patches, and for each patch we get an id, a distance to boundary, and parameterized borders.
//
// Trees and houses placement:
//
//   - Patches ids is used to decide what combination of things goes on the patch (trees, bushes, farms, walls, lawn...)
//   - There are 3 layers of cellular placement for trees, bushes, and farms.
//     - Bushes are too close to each other and must be soft blended, but 3x3 search is no-no so we do a "4 or 5" neighbours search (we only consider checkboard black cells).
//     - For farms and trees we use randomly decimated jittered grid and actually only consider the current cell we are in, and hack marching to death to compensate.
//   - Modeling:
//     - Trees leaves volume have a base shape done with 2 spheres soft blended, then distored by 2 layers of packed 3d spheres tiling to blobify the leaves volume, and then some fine noise distortion on the surface.
//       The use of densely packed sphere tiling is inspired by @Shane's Cellular Tiling https://www.shadertoy.com/view/4scXz2
//     - Farms are randomized with gable and hipped roof, chimneys and colors very vaguely inspired by pictures of Devon.
//
// Marching:
//
//   - For patches, marching uses ghost steps nearby patch boundaries so that we don't check neighbour patches objects, only the patch we are in.
//   - For trees and farms too, we force the raymarch to take ghost steps along their cell borders for x1 sdf eval.
//     - This ghost point machinery is hacky and not perfect (esp on patches boundary where we don't have clean intersections) but still helps.
//   - Because of all the cellular evals going on, to save height evals we use taylor expansion of the heightfield on local neiborhood.
//   - Despite above efforts I had to resort to reprojection and still perf isn't great. 
//     Blurring the noise with reprojection also helps hide the general noisy lameness and gives better colors.
//
// Clouds are volumetric but baked in a spheremap at first frame and assumed distant.
// Also had to turn view trace/shadow trace/scene gradient/cellular evals into loops to help compile time on the website, sometimes at the expense of runtime perfs.
// As always some code, techniques, ideas from @iq, @Dave_Hoskins, @Shane, @FabriceNeyret2 are used in various places, 
// this shader also uses some spherical gaussian code from Matt Pettineo 
// (see comment for links to references).

#define PI 3.141592654 // He does love his numbers
#define FLT_MAX 1000000.0

#define SQR2 1.414213562
#define SQR3 1.732050807

#define RED  vec3( 1, 0, 0 )
#define GREEN vec3( 0, 1, 0 )
#define BLUE vec3( 0, 0, 1 )
#define WHITE vec3( 1, 1, 1 )
#define BLACK vec3( 0, 0, 0 )
#define MAGENTA vec3( 1, 0, 1 )
#define YELLOW vec3( 1, 1, 0 )
#define AZURE vec3( 0.0, 0.5, 1.0 )

#define A_FEW_FUNC(type) \
type saturate( type x ) { return clamp( x, type(0.0), type(1.0) ); } \
type smoothstep_unchecked( type x ) { return ( x * x ) * ( type(3.0) - x * 2.0 ); } \
type smoothstep_unchecked_d( type x ) { return 6.0 * x * ( type(1.0) - x ); }

A_FEW_FUNC( float )
A_FEW_FUNC( vec2 )
A_FEW_FUNC( vec3 )
A_FEW_FUNC( vec4 )

float exp_decay( float x ) { return 1. - exp( -x ); }
// cubic bump that goes through (0,0)->(1,0)
// slope at x=0 is df0
// slope at x=1 is df1
float cubicbump( float x, float df0, float df1 ) { float a = df1 + df0; float c = df0; float b = -a - c; return x * ( x * ( x * a + b ) + c ); }
float smoothbump( float a, float r, float x ) { return 1.0 - smoothstep_unchecked( min( abs( x - a ), r ) / r ); }
// like smoothstep, but takes a center and a radius instead
float smoothstep_c( float x, float c, float r ) { return smoothstep( c - r, c + r, x ); }
// centered at 0
float smoothband( float x, float r, float raa ) { return 1. - smoothstep_c( abs( x ), r, raa ); }
// range s,e
float smoothband( float x, float s, float e, float raa ) { return smoothband( x - ( e + s ) * 0.5, ( e - s ) * 0.5, raa ); }
vec2 perp( vec2 v ) { return vec2( -v.y, v.x ); }
float calc_angle( vec2 v ) { return atan( v.y, v.x ); } // return range -pi,pi
float calc_angle( vec2 a, vec2 b ) { return calc_angle( vec2( dot( a, b ), dot( perp( a ), b ) ) ); }
float contrast( float x, float s ) { return ( x - 0.5 ) * s + 0.5; }
vec3 contrast( vec3 x, vec3 s ) { return ( x - 0.5 ) * s + 0.5; }
float lensqr( vec2 v ) { return dot( v, v ); }
float lensqr( vec3 v ) { return dot( v, v ); }
float lensqr( vec4 v ) { return dot( v, v ); }
float pow2( float x ) { return x * x; }
vec3 pow2( vec3 x ) { return x * x; }
vec4 pow2( vec4 x ) { return x * x; }
// variant of exp/log soft min and max that save a few instructions
float smin_exp2( float a, float b, float k ) { return -log2( exp2( -k * a ) + exp2( -k * b ) ) / k; }
float smax_exp2( float a, float b, float k ) { return -smin_exp2( -a, -b, k ); }
// http://iquilezles.org/www/articles/smin/smin.htm
float smin_pol( float a, float b, float k ) { float h = clamp( 0.5f + 0.5f * ( b - a ) / k, 0.0f, 1.0f ); return mix( b, a, h ) - k * h * ( 1.0 - h ); }
float smax_pol( float a, float b, float k ) { return -smin_pol( -a, -b, k ); }
float powerful_scurve( float x, float p1, float p2 ) { return pow( 1.0 - pow( 1.0 - clamp( x, 0.0, 1.0 ), p2 ), p1 ); }
float maxcomp( float x ) { return x; }
float maxcomp( vec2 v ) { return max( v.x, v.y ); }
float maxcomp( vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float min_( float x, float y, float z ) { return min( min( x, y ), z ); }
float max_( float x, float y, float z ) { return max( max( x, y ), z ); }
float nearest( float x ) { return floor( 0.5 + x ); }
float nearest( float x, float spacing ) { return floor( 0.5 + x / spacing ) * spacing; }
vec2 nearest( vec2 x, vec2 spacing ) { return floor( vec2( 0.5 ) + x / spacing ) * spacing; }
float sum( vec4 v ) { return v.x + v.y + v.z + v.w; }
float safe_acos( float x ) { return acos( clamp( x, -1., 1. ) ); }

// project this on line (O,d), d is assumed to be unit length
#define PROJECT_ON_LINE1(type) \
type project_on_line( type P, type O, type d ) { return O + d * dot( P - O , d ) / dot(d, d ); } \
type project_on_plane( type P, type O, type n ) { return P - n * dot( P - O , n ) / dot(n, n ); } \

PROJECT_ON_LINE1( vec2 )
PROJECT_ON_LINE1( vec3 )

// http://www.iquilezles.org/www/articles/distance/distance.htm
// fast and nice in most cases
#define FAST_SIGNED_DISTANCE_TO_FUNC_11_ARG( a_signed_distance, a_p, a_func, a_arg, a_epsilon ) \
{ \
    vec2 _p = a_p; \
    float _epsilon = a_epsilon; \
    float _y = a_func( _p.x, a_arg ); \
    float _y1 = a_func( _p.x + _epsilon, a_arg ); \
    float _ddy = ( _y1 - _y ) * ( 1. / _epsilon ); \
    a_signed_distance = ( _p.y - _y ) / sqrt( 1. + _ddy * _ddy ); \
}

// this macro returns:
//  - the vector to the closest point on a curve (the length of which gives a better distance than FAST_SIGNED_DISTANCE_TO_FUNC_11) 
//  - the tangent at that closest point
// http://www.geometrie.tugraz.at/wallner/sproj.pdf
// input:
//  a_p               eval at pos
//  a_funcd           is the R->R function to evaluate, first guess (iteration start point) is vec2(a_p.x,a_func(a_p.x))
//                    a_funcd can be a macro, it seems
//                    returns the function value at t in .x and the derivative at t in .y
//  a_funcd_arg       an argument passed to a_func
//  a_cheap           num iterations, 2 should be enough, a_cheap==true only does 1 iteration
//
// output:
//  a_ret             a_ret.xy is vector to closest point on curve
//                    a_ret.zw is the derivative (tangent at the closest point)
//
// note: we could get the sign from the first iteration

#define CLOSEST_POINT_TANGENT_TO_FUNCD_11_ARG_CHEAP( a_ret, a_p, a_funcd, a_funcd_arg, a_cheap ) \
{ \
    vec2 _p = a_p.xy, _c, _dc, _ev; \
    float _t = _p.x; /* t0, could be a parameter if the user knows better */ \
    _ev = a_funcd( _t, a_funcd_arg ); \
    _c = vec2( _t, _ev.x ); \
    _dc = vec2( 1.0, _ev.y ); /*important: 1 in x!*/ \
    if ( !(a_cheap) ) /* IMPORTANT: if num iteration is 2, an if test can behave much better than for loop */ \
    { \
        /*#if 0*/ \
        /*vec2 _q = project_on_line( _p.xy, _c, _dc );*/ \
        /*_t += dot( _dc, _q - _c ) / dot( _dc, _dc );*/ \
        /* simplifies to: */ \
        /*#else*/ \
        _t += dot( _p.xy - _c, _dc ) / dot( _dc, _dc ); \
        /*#endif*/ \
        _ev = a_funcd( _t, a_funcd_arg ); \
        _c = vec2( _t, _ev.x ); \
        _dc = vec2( 1.0, _ev.y ); /*important: 1 in x!*/ \
    } \
    a_ret = vec4( _c - _p, _dc ); \
}

vec3 transform_vector( mat4 m, vec3 v ) { return ( m * vec4( v, 0.0 ) ).xyz ; }

struct bounds2 { vec2 pmin; vec2 pmax; };
bounds2 mkbounds_unchecked( vec2 amin, vec2 amax ) { bounds2 ret; ret.pmin = amin; ret.pmax = amax; return ret; }

#define REPEAT_FUNCTIONS( type ) \
type repeat( type x, type len ) { return len * fract( x * ( type( 1.0 ) / len ) ); }\
type repeat_mirror( type x, type len ) { return len * abs( type( -1.0 ) + 2.0 * fract( ( ( x * ( type( 1.0 ) / len ) ) - type( -1.0 ) ) * 0.5 ) ); }

REPEAT_FUNCTIONS( float )
REPEAT_FUNCTIONS( vec2 )

// badly antialiased stripes
// r is the half width of the stripes
// raa is the half size of the edge/aa smoothstep (ex: pixel_size)
// period is the distance between 2 consecutive stripes

float stripes( float x, float period, float r, float raa ) { return smoothstep( r + raa, r - raa, repeat_mirror( x, period * 0.5 ) ); }
vec2 stripes( vec2 x, vec2 period, vec2 r, vec2 raa ) { return smoothstep( r + raa, r - raa, repeat_mirror( x, period * 0.5 ) ); }

// triangular sin waves - you can drop in as a replacement for sin to get polygonized looks
float tri_sin( float x ) { return (abs(fract((x-PI*0.5)/(PI*2.))-0.5)-0.25)*4.0; }

// hash functions from David Hoskins's https://www.shadertoy.com/view/4djSRW

// Hash without Sine
// MIT License...
/* Copyright (c)2014 David Hoskins.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

float hash11(float p) { p = fract(p * .1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx ) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float hash13(vec3 p3) { p3 = fract(p3 * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 hash22(vec2 p) { vec3 p3 = fract(vec3(p.xyx ) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yzx +33.33); return fract((p3.xx +p3.yz )*p3.zy ); }
vec2 hash23(vec3 p3) { p3 = fract(p3 * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yzx +33.33); return fract((p3.xx +p3.yz )*p3.zy ); }
vec3 hash31(float p) { vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yzx +33.33); return fract((p3.xxy +p3.yzz )*p3.zyx ); }
vec3 hash32(vec2 p) { vec3 p3 = fract(vec3(p.xyx ) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yxz +33.33); return fract((p3.xxy +p3.yzz )*p3.zyx ); }
vec3 hash33(vec3 p3) { p3 = fract(p3 * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yxz +33.33); return fract((p3.xxy + p3.yxx )*p3.zyx ); }

//###############################################################################

// iq's function munged for vec4, used in city shader...
// https://www.shadertoy.com/view/XlXcW4 note: source has changed since then...

vec4 hash42_( ivec2 x0 )
{
    uint k = 1103515245U;  // GLIB C
    uvec4 x = uvec4( x0, x0 * 0x8da6b343 );
    x = (( x >> 13U ) ^ x.yzwx ) * k;
    x = (( x >> 13U ) ^ x.zwxy ) * k;
//  x = (( x >> 13U ) ^ x.wxyz ) * k; // can't really tell the difference
    return vec4( x ) * ( 1.0 / float( 0xffffffffU ));
}

// integer hashes
// https://www.shadertoy.com/view/4tXyWN iq

float hash1u2_4tXyWN( uvec2 x )
{
    uvec2 q = 1103515245U * ( ( x >> 1U ) ^ ( x.yx  ) );
    uint  n = 1103515245U * ( ( q.x  ) ^ ( q.y >> 3U ) );
    return float( n ) * ( 1.0 / float( 0xffffffffU ) );
}

// https://nullprogram.com/blog/2018/07/31/ Chris Wellons
// https://www.shadertoy.com/view/WttXWX via Fabrice

uint lowbias32(uint x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
float hash11u_lowbias32( uint x ) { return float( lowbias32( x ) ) / float( 0xffffffffU ); }

#define V30 vec2( 0.866025403, 0.5 )
#define V45 vec2( 0.707106781, 0.707106781 )
#define V60 vec2( 0.5, 0.866025403 )

// return a unit vector, or an angle (it's the same thing)
vec2 unit_vector2( float angle ) { return vec2( cos( angle ), sin( angle ) ); }
// note that if point p is also a unit vector, rotate_with_unit_vector returns the same as doing unit_vector2 on the sum of the angles (obvious but)
vec2 rotate_with_unit_vector( vec2 p, vec2 cs ) { return vec2( cs.x * p.x - cs.y * p.y, cs.y * p.x + cs.x * p.y ); }
vec2 rotate_with_angle( vec2 p, float a_angle ) { return rotate_with_unit_vector( p, unit_vector2( a_angle ) ); }

// theta is angle with the z axis, range [0,pi].
// phi is angle with x vectors on z=0 plane, range [0,2pi].
// theta_vec is the unit vector for angle theta
// phi_vec is the unit vector for angle phi
vec3 zup_spherical_coords_to_vector( vec2 theta_vec, vec2 phi_vec ) { return vec3( theta_vec.y * phi_vec, theta_vec.x ); }
vec3 zup_spherical_coords_to_vector( float theta, float phi ) { return zup_spherical_coords_to_vector( unit_vector2( theta ), unit_vector2( phi ) ); }
vec3 zup_spherical_coords_to_vector( vec2 theta_phi ) { return zup_spherical_coords_to_vector( theta_phi.x, theta_phi.y ); }

// note: n.xy==0 is undefined for phi, pleae handle in caller code
vec2 vector_to_zup_spherical_coords( vec3 n )
{
    float theta = safe_acos( n.z ); // note: vectors normalized with normalize() are not immune to -1,1 overflow which cause nan in acos
    float phi = calc_angle( n.xy  );
    return vec2( theta, phi );
}

vec3 yup_spherical_coords_to_vector( vec2 theta, vec2 phi ) { return zup_spherical_coords_to_vector( theta, phi ).yzx ; }
vec3 yup_spherical_coords_to_vector( float theta, float phi ) { return yup_spherical_coords_to_vector( unit_vector2( theta ), unit_vector2( phi ) ); }

mat4 yup_spherical_coords_to_matrix( vec2 theta, vec2 phi )
{
    vec3 y = yup_spherical_coords_to_vector( theta, phi );
    vec3 z = yup_spherical_coords_to_vector( perp( theta ), phi ); // note: perp(theta) = unit_vector2(theta+PI*0.5)
    vec3 x = cross( y, z );
    return ( mat4( vec4( x, 0.0 ), vec4( y, 0.0 ), vec4( z, 0.0 ), vec4( 0, 0, 0, 1 ) ) );
}

mat4 yup_spherical_coords_to_matrix( float theta, float phi ) {  return yup_spherical_coords_to_matrix( unit_vector2( theta ), unit_vector2( phi ) ); }

mat4 z_rotation( float angle ) { vec2 v = unit_vector2( angle ); return mat4( vec4( v.x, v.y, 0.0, 0.0 ), vec4( -v.y, v.x, 0.0, 0.0 ), vec4( 0, 0, 1, 0 ), vec4( 0, 0, 0, 1 ) ); }

mat3 mat3_rotation_x( vec2 v ) { return mat3( vec3( 1, 0, 0 ), vec3( 0, v.x, v.y ), vec3( 0, -v.y, v.x ) ); }
mat3 mat3_rotation_z( vec2 v ) { return mat3( vec3( v.x, v.y, 0 ), vec3( -v.y, v.x, 0 ), vec3( 0, 0, 1 ) ); }

// icdf for pdf a*exp(-a*x) (wikipedia Exponential distribution)
float ed_icdf( float u, float a ) { return -log( 1. - u ) * ( 1.0 / a ); }

#define INDEXHASHOFFSET3 vec3( 137.0, 53.0, 173 )

#define LAYERED1(func,p,args) ((func(p*1.0,args)*0.5)                                                                                               *(1.0/(0.5)))
#define LAYERED2(func,p,args) ((func(p*1.0,args)*0.5+func(p*2.0,args)*0.25)                                                                         *(1.0/((0.5+0.25))))
#define LAYERED4(func,p,args) ((func(p*1.0,args)*0.5+func(p*2.0,args)*0.25+func(p*4.0,args)*0.125+func(p*8.0,args)*0.0625)                          *(1.0/((0.5+0.25+0.125+0.0625))))
#define LAYERED5(func,p,args) ((func(p*1.0,args)*0.5+func(p*2.0,args)*0.25+func(p*4.0,args)*0.125+func(p*8.0,args)*0.0625+func(p*16.0,args)*0.03125)*(1.0/((0.5+0.25+0.125+0.0625+0.03125))))

float noise12( vec2 p, bool use_smooth )
{
    p -= 0.5;

    vec2 p00 = floor( p );
    vec2 p10 = p00 + vec2( 1, 0 );
    vec2 p01 = p00 + vec2( 0, 1 );
    vec2 p11 = p00 + vec2( 1, 1 );

    float v00 = hash12( p00 );
    float v10 = hash12( p10 );
    float v01 = hash12( p01 );
    float v11 = hash12( p11 );

    vec2 f = fract( p ); // p-p00 but beware large values...

    if ( use_smooth ) f = smoothstep_unchecked( f );

    float A = v10 - v00;
    float B = v01 - v00;
    float C = ( v11 - v01 ) - A;
    float D = v00;

//  return mix( mix( v00, v10, f.x ), mix( v01, v11, f.x ), f.y ),
    return A * f.x + B * f.y + C * f.x * f.y + D;
}

float noise13( vec3 p, bool use_smooth )
{
    p -= 0.5;

    vec3 p000 = floor( p ) + INDEXHASHOFFSET3;
    vec3 p100 = p000 + vec3( 1, 0, 0 );
    vec3 p010 = p000 + vec3( 0, 1, 0 );
    vec3 p110 = p000 + vec3( 1, 1, 0 );
    vec3 p001 = p000 + vec3( 0, 0, 1 );
    vec3 p101 = p000 + vec3( 1, 0, 1 );
    vec3 p011 = p000 + vec3( 0, 1, 1 );
    vec3 p111 = p000 + vec3( 1, 1, 1 );

    float v000 = hash13( p000 );
    float v100 = hash13( p100 );
    float v010 = hash13( p010 );
    float v110 = hash13( p110 );
    float v001 = hash13( p001 );
    float v101 = hash13( p101 );
    float v011 = hash13( p011 );
    float v111 = hash13( p111 );

    vec3 f = fract( p ); // bilinear

    if ( use_smooth ) f = smoothstep_unchecked(f);
    
    return mix( mix( mix( v000, v100, f.x ),
                     mix( v010, v110, f.x ), f.y ),
                mix( mix( v001, v101, f.x ),
                     mix( v011, v111, f.x ), f.y ), f.z );
}

vec3 noise33( vec3 p, bool use_smooth )
{
    p -= 0.5;

    vec3 p000 = floor( p ) + INDEXHASHOFFSET3;
    vec3 p100 = p000 + vec3( 1, 0, 0 );
    vec3 p010 = p000 + vec3( 0, 1, 0 );
    vec3 p110 = p000 + vec3( 1, 1, 0 );
    vec3 p001 = p000 + vec3( 0, 0, 1 );
    vec3 p101 = p000 + vec3( 1, 0, 1 );
    vec3 p011 = p000 + vec3( 0, 1, 1 );
    vec3 p111 = p000 + vec3( 1, 1, 1 );

    vec3 v000 = hash33( p000 );
    vec3 v100 = hash33( p100 );
    vec3 v010 = hash33( p010 );
    vec3 v110 = hash33( p110 );
    vec3 v001 = hash33( p001 );
    vec3 v101 = hash33( p101 );
    vec3 v011 = hash33( p011 );
    vec3 v111 = hash33( p111 );

    vec3 f = fract( p ); // bilinear

    if ( use_smooth ) f = smoothstep_unchecked(f); // looks sharper in 3d

    return mix( mix( mix( v000, v100, f.x ),
                     mix( v010, v110, f.x ), f.y ),
                mix( mix( v001, v101, f.x ),
                     mix( v011, v111, f.x ), f.y ), f.z );
}

float enoise13( vec3 p, float a, bool use_smooth )
{
    p -= 0.5;

    vec3 p000 = floor( p ) + INDEXHASHOFFSET3;
    vec3 p100 = p000 + vec3( 1, 0, 0 );
    vec3 p010 = p000 + vec3( 0, 1, 0 );
    vec3 p110 = p000 + vec3( 1, 1, 0 );
    vec3 p001 = p000 + vec3( 0, 0, 1 );
    vec3 p101 = p000 + vec3( 1, 0, 1 );
    vec3 p011 = p000 + vec3( 0, 1, 1 );
    vec3 p111 = p000 + vec3( 1, 1, 1 );

    vec2 h000 = hash23( p000 );
    vec2 h100 = hash23( p100 );
    vec2 h010 = hash23( p010 );
    vec2 h110 = hash23( p110 );
    vec2 h001 = hash23( p001 );
    vec2 h101 = hash23( p101 );
    vec2 h011 = hash23( p011 );
    vec2 h111 = hash23( p111 );

    float v000 = ed_icdf( h000.x, a ) * h000.y;
    float v100 = ed_icdf( h100.x, a ) * h100.y;
    float v010 = ed_icdf( h010.x, a ) * h010.y;
    float v110 = ed_icdf( h110.x, a ) * h110.y;
    float v001 = ed_icdf( h001.x, a ) * h001.y;
    float v101 = ed_icdf( h101.x, a ) * h101.y;
    float v011 = ed_icdf( h011.x, a ) * h011.y;
    float v111 = ed_icdf( h111.x, a ) * h111.y;

    vec3 f = fract( p ); // bilinear

    if ( use_smooth ) f = smoothstep_unchecked(f); // looks sharper in 3d

    return mix( mix( mix( v000, v100, f.x ),
                     mix( v010, v110, f.x ), f.y ),
                mix( mix( v001, v101, f.x ),
                     mix( v011, v111, f.x ), f.y ), f.z );
}

// prefix meaning: 
//  e stands for exponential distribution
//  s stands for smoothstep interpolation
float snoise12_( vec2 p, float args ) { return noise12( p, true ); }
float snoise13_( vec3 p, float args ) { return noise13( p, true ); }
vec3 noise33_( vec3 p, float args ) { return noise33( p, false ); }
float enoise13_( vec3 p, float a ) { return enoise13( p, a, false ); }
float sfbm1_12( vec2 p ) { return LAYERED1( snoise12_, p, -1.0); }
float sfbm2_13( vec3 p ) { return LAYERED2( snoise13_, p, -1.0); }
float sfbm2_13_leaf( vec3 p ) { return (noise13(p*0.8,true)+noise13(p*4.0,true)*0.6)/1.5;}
float sfbm2_12( vec2 p ) { return LAYERED2( snoise12_, p, -1.0); }
float efbm4_13( vec3 p, float arg ) { return LAYERED4( enoise13_, p, arg); }
vec3 sfbm4_33( vec3 p ) { return LAYERED5( noise33_, p, -1.0); }

struct Ray { vec3 o; vec3 d;  };

Ray mkray( vec3 o, vec3 d ) { Ray tmp; tmp.o = o; tmp.d = d; return tmp; }

vec3 get_view_dir( vec2 normalized_pos, float aspect, float tan_half_fovy_rcp )
{
    return normalize( vec3( normalized_pos.x * aspect, normalized_pos.y, -tan_half_fovy_rcp ) ); // note: looking down z
}

// same as get_view_ray_old but without a znear
// note that we pass the reciprocal of tan_half_fovy
// normalized_pos is (-1,1-)->(1,1)
Ray get_view_ray2( vec2 normalized_pos, float aspect, float tan_half_fovy_rcp, mat4 camera )
{
    return mkray( camera[3].xyz , transform_vector( camera, get_view_dir( normalized_pos, aspect, tan_half_fovy_rcp ) ) );
}

mat4 lookat( vec3 eye, vec3 center, vec3 up )
{
    vec3 z = normalize( eye - center );
    vec3 x = normalize( cross( up, z ) );
    vec3 y = cross( z, x );
    return mat4( vec4( x, 0.0 ), vec4( y, 0.0 ), vec4( z, 0.0 ), vec4( eye, 1.0 ) );
}

vec2 sphere_trace( Ray ray, float radius, vec3 center )
{
    vec3 O = ray.o;
    vec3 d = ray.d;
    float tp = dot( center - O, d ); // O + d * tp = center projected on line (O,d)
    float h_sqr = lensqr( ( O + d * tp ) - center );
    float radius_sqr = radius * radius;
    if ( h_sqr > radius_sqr ) return vec2( FLT_MAX, FLT_MAX ); // ray missed the sphere
    float dt = sqrt( radius_sqr - h_sqr ); // distance from P to In (near hit) and If (far hit)
    return vec2( tp - dt, tp + dt ); // record 2 hits In, If
}

float plane_trace( vec3 ray_o, vec3 ray_d, vec3 base, vec3 n, float epsilon ) { float ddotn = dot( ray_d, n ); return abs( ddotn ) > epsilon ? dot( base - ray_o, n ) / ddotn : FLT_MAX; }
float plane_trace( Ray ray, vec3 base, vec3 n, float epsilon ) { float ddotn = dot( ray.d, n ); return abs( ddotn ) > epsilon ? dot( base - ray.o, n ) / ddotn : FLT_MAX; }
float plane_trace_z( Ray ray, float base, float epsilon ) { return abs( ray.d.z ) > epsilon ? ( base - ray.o.z ) / ray.d.z : FLT_MAX; }
// d is a unit direction, ray starts at 0,0,0 base is plane position along z, this is just a division...
float plane_trace_z( vec3 d, float base, float epsilon ) { return abs( d.z ) > epsilon ? base / d.z : FLT_MAX; }

// build a little quadric so that y'(0)=0, y(r)=r, y'(r)=1 here
float her2( float x, float r ) { return 0.5 * ( ( 1.0 / r ) * x * x + r ); }
// smooth bevel (like a soft_abs function)
float curved_max_vfunc_weld_quadric( float x, float r ) { x = abs( x ); return x > r ? x : her2( x, r ); }
// max
float opI( float d1, float d2 ) { return max( d1, d2 ); }
float opI_soft2( float a, float b, float k ) { return smax_exp2( a, b, k ); }
float opI_soft2_pol( float a, float b, float k ) { return smax_pol( a, b, k ); }
float opI_weld_quadric( float a, float b, float r ) { float c = ( a + b ) * 0.5; return c + curved_max_vfunc_weld_quadric( a - c, r ); }
// min(a,b) = -max(-a,-b)
float opU( float d1, float d2 ) { return -max( -d1, -d2 ); }
float opU_soft2_pol( float a, float b, float k ) { return -opI_soft2_pol( -a, -b, k ); }
float opU_weld_quadric( float a, float b, float r ) { return -opI_weld_quadric( -a, -b, r ); }
float opS( float d1, float d2 ) { return max( -d2, d1 );}
float opS_soft2( float a, float b, float k ) { return opI_soft2( -b, a, k ); }
float opI( float d1, float d2, float d3 ) { return max( max( d1, d2 ), d3 ); }

// r can be zero
float sd_bounds_range_round( vec2 p, vec2 mi, vec2 ma, float r )
{
    vec2 h = ( ma - mi ) * 0.5;
    p = abs( p - ( mi + ma ) * 0.5 );
    vec2 c = h - r;
    float mask = maxcomp( step( c, p ) );
    return mix( maxcomp( p - c ), length( max( p - c, vec2( 0.0 ) ) ), mask ) - r;
}

// r can be zero
float sd_bounds_range_round( vec3 p, vec3 mi, vec3 ma, float r )
{
    vec3 h = ( ma - mi ) * 0.5;
    p = abs( p - ( mi + ma ) * 0.5 );
    vec3 c = h - r;
    float mask = maxcomp( step( c, p ) );
    return mix( maxcomp( p - c ), length( max( p - c, vec3( 0.0 ) ) ), mask ) - r;
}

float sd_bounds_half_size( float p, float h ) { p = abs( p ) - h; return p; }
float sd_bounds_half_size( vec2 p, vec2 h ) { p = abs( p ) - h; return opI( p.x, p.y ); }
float sd_bounds_half_size( vec3 p, vec3 h ) { p = abs( p ) - h; return opI( p.x, p.y, p.z ); }
float sd_bounds_range( vec2 p, vec2 mi, vec2 ma ) { vec2 hmi = mi * 0.5; vec2 hma = ma * 0.5; return sd_bounds_half_size( p - ( hma + hmi ), hma - hmi ); }
// those bounds repeat might be good after all, since they centering and lead to a correct repeat...
float sd_bounds_range( float p, float mi, float ma ) { return sd_bounds_half_size( p - ( ( ma + mi ) * 0.5 ), ( ma - mi ) * 0.5 ); }
float sd_bounds_range( vec3 p, vec3 mi, vec3 ma ) { return sd_bounds_half_size( p - ( ( ma + mi ) * 0.5 ), ( ma - mi ) * 0.5 ); }

float sd_sphere( vec3 p, vec3 center, float radius ) { return length( p - center ) - radius; }
float sd_sphere( vec2 p, vec2 center, float radius ) { return length( p - center ) - radius; }

// iq's https://www.shadertoy.com/view/Xds3zN modified for z up
float sdCylinder( vec3 p, vec2 h )
{
    vec2 d = abs( vec2( length( p.xy ),p.z)) - h;
    return min( max( d.x, d.y ), 0.0 ) + length( max( d, vec2( 0.0 ) ) );
}

// internal function in packed_spheres_tiling3d
float packed_spheres_tiling3d_internal_layer( vec3 p )
{
    vec2 rh = vec2( 1.0, SQR3 ); // r=1  (normally h = spacing*0.5 * SQR3, and here spacing=2)
    vec2 c = rh * 2.0; // cell size for each row
    vec2 i1 = floor( p.xy  / c );
    vec2 i2 = floor( ( p.xy  - rh ) / c );
    return min(
        length( p - vec3( ( i1 + vec2( 0.5 ) ) * c, 0 ) ) - 1.0f,
        length( p - vec3( ( i2 + vec2( 0.5 ) ) * c + rh, 0 ) ) - 1.0f ); // second row offset by rh
}

// sdf of packed spheres of radius 1 (just add to distance for smaller radius...)
float packed_spheres_tiling3d( vec3 p )
{
    vec3 p0 = p;
    float h = SQR3; // height of equilateral triangle of edge len 1+1=2
    float b = ( 1.0 - h * h ) / ( -2.0 * h );
    float c = h - b;
    float h3 = sqrt( 2.0 * 2.0 - c * c ); // height of tetrahedra, also the spacing between layers, also the half period of each layer
    p = p0;
    p.z -= ( floor( ( p.z - ( -h3 ) ) / ( 2.0 * h3 ) ) + 0.5 ) * ( 2.0 * h3 ) - h3; // repeat layer
    float d1 = packed_spheres_tiling3d_internal_layer( p );
//  return d1;
    p = p0;
    p.y += h - b; // offset to overlap centers of first layer exactly
    p.z -= ( floor( ( p.z - 0.0 ) / ( 2.0 * h3 ) ) + 0.5 ) * ( 2.0 * h3 ) - 0.0; // repeat layer
    float d2 = packed_spheres_tiling3d_internal_layer( p );
//  return d2;
//  return opU( d1, d2 );
    return min( d1, d2 );
}

// r is sphere radius, distance between 2 spheres is spacing
// r must be < spacing/2
float packed_spheres_tiling3d( vec3 p, float r, float spacing )
{
    float s = spacing * 0.5; // packed sphere radius
    return packed_spheres_tiling3d( p * ( 1.0 / s ) ) * s + ( s - r );
}

// r is sphere radius, distance between 2 spheres is 2*r
float packed_spheres_tiling3d( vec3 p, float r )
{
    return packed_spheres_tiling3d( p * ( 1.0 / r ) ) * r;
}

vec3 tonemap_reinhard( vec3 x ) { return x / ( 1. + x ); }
// mentioned in http://resources.mpi-inf.mpg.de/tmo/logmap/
vec3 gamma_correction_itu( vec3 L ) { return mix( 4.5061986 * L, 1.099 * pow( L, vec3( 0.45 ) ) - 0.099, step( vec3( 0.018 ), L ) ); }

// the couple of following functions are copied from Matt Pettineo's spherical gaussian article, 
// I liked the soft look and ease of use of SG and ended up keeping to the end
// https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-2-spherical-gaussians-101/

struct SG { vec3 Amplitude; vec3 Axis; float Sharpness; };

// approximate integral on omega of sg(v)dv
vec3 ApproximateSGIntegral( in SG sg ) { return 2. * PI * ( sg.Amplitude / sg.Sharpness ); }

SG CosineLobeSG( in vec3 direction )
{
    SG cosineLobe;
    cosineLobe.Axis = direction;
    cosineLobe.Sharpness = 2.133f;
    cosineLobe.Amplitude = vec3( 1.17f );
    return cosineLobe;
}

// https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-3-diffuse-lighting-from-an-sg-light-source/

vec3 SGIrradianceFitted( in SG lightingLobe, in vec3 normal )
{
    float muDotN = dot( lightingLobe.Axis, normal );
    float lambda = lightingLobe.Sharpness;

    float c0 = 0.36f;
    float c1 = 1.0f / ( 4.0f * c0 );

    float eml  = exp( -lambda );
    float em2l = eml * eml;
    float rl   = 1.0 / lambda;

    float scale = 1.0f + 2.0f * em2l - rl;
    float bias  = ( eml - em2l ) * rl - em2l;

    float x  = sqrt( 1.0f - scale );
    float x0 = c0 * muDotN;
    float x1 = c1 * x;

    float n = x0 + x1;

    float y = saturate( muDotN );
    if ( abs( x0 ) <= x1 ) y = n * n / x;

    float result = scale * y + bias;

    return result * ApproximateSGIntegral( lightingLobe );
}

// what am I doing
vec3 SGDiffuseFitted( in SG lightingLobe, in vec3 normal, vec3 albedo )
{
    vec3 brdf = albedo / PI;
    return SGIrradianceFitted( lightingLobe, normal ) * brdf;
}

// BUFFER_MODE
// 0: normal size (ugly but use to check artifacts in high def, also cloudless)
// 2: halfsize + history reprojection... blurry-but-AA type of thing
#define BUFFER_MODE 2

// iChannel0 is always Buf A
// iChannel1 is always Buf B

// note: glsl doesn't let us write if (1), so we write if (_1), note that occasionally compiler may fail to treat that as a static branch
#define _1 true
#define _0 false

// skip space faster above bushes (and maybe other things if proves useful)
// notes:
//  - at higher res we can see that causes artifacts, see the bit that sets
//    eval.terrain_warp and tweak hack constants there
//  - this causes artifacts on trees's tracing... close to patch borders maybe? (messes with shadows sometimes)
//  - this messes with bush shadows even when bush surface seems unchanged (shrinks shadows a bit)
#define TERRAIN_WARP

#define FORCE_LOOP +min(0,aFrame)
#define FORCE_LOOPF +min(0.0,float(aFrame))

#define SHADOW_TINT_SATURATION 0.45 // [0,1]

#define AO _0 // nice term, visible extra cost, but doesn't contribute much, so disable 
#define SHADOWS _1

#define SUN _1
#define SUN_GLARE _0 // could be interesting but needs more work

#define GROUND_OCCLUSION _1 // first sdf is ground's base level so we get occlusion
#define PATH _1 // cost peanuts
#define GRASS _1 // cost peanuts
#define BUSH _1 // a bit expensive
#define TREE _1 // super expensive
#define FARM _1 // cost a little bit
#define BRICKWALL _1 // cost a little bit

#define TAYLOR_HEIGHT_BUSH _1
#define TAYLOR_HEIGHT_TREE _1
#define TAYLOR_HEIGHT_FARM _0 // object is too large, causes a little bit too much distortion

#define GHOST_STEPS_PATCH _1
#define GHOST_STEPS_TREE _1 // visible extra cost, fixes the trees, but we start hitting iteration limit
#define GHOST_STEPS_FARM _1 // little cost (large cells) and fixes farms, hits a little bit more max iterations far away
// about GHOST_MAX_DIST:
// introduced a second distance for bird views
// having 2 thresh lets us fix bird view, but they get more expensive
#define GHOST_MAX_DIST_FROM_T0 7.0
#define GHOST_MAX_DIST_ABS 25.0
// fixes weird grain that happens when using second threshold that goes further away... hack hack...
#define GHOST_EPS 0.0001

#define DISPLACE_PATH _1 // a bit expensive, actually
#define DISPLACE_PATH_AMPL 0.002

#define DISPLACE_PINE _1
#define DISPLACE_TREE _1
#define DISPLACE_BUSH _1
#define DISPLACE_BUSH_FREQ 8.0

// CLOUD_MODE
//  0: no cloud
//  1: trace cloud per pixel (for debug)
//  2: cache trace cloud in B texture every frame and use that (for debug)
// >3: cache trace cloud in B texture only first frame and use that (same cost as no cloud)
#define CLOUD_MODE 3

// the wind doesn't have much amplitude, and movement isn't smooth enough, but reprojection hides that a bit...
#define WIND_BUSH _1
#define WIND_TREE_AND_PINES _1

#define FARM_WINDOWS _1

#define SOFT_BLEND_BUSH _1
#define SQUARE_BUSH_TEST _1 // some farms have nicely cut bushes arounds them

#define MATID_NONE    0.0
#define MATID_BUSH    1.0
#define MATID_TRUNK   2.0
#define MATID_HOUSE   3.0
#define MATID_ROOF    4.0
#define MATID_PINE    5.0
#define MATID_BRICKWALL   6.0
#define MATID_GRASS   7.0
#define MATID_GROUND  8.0
#define MATID_TREE    9.0
#define MATID_WINDOW 10.0
#define MATID_HOUSE_BOTTOM 11.0

#define BROWN (vec3(133,84,57)/255.0)
#define PATH_COLOR BROWN
#define PATH_COLOR2 (BROWN*0.8)
#define COLOR_BUSH1 (0.8*vec3(0.07,0.3,0.05))
#define COLOR_BUSH2 (0.55*vec3(0.12,0.6,0.2))
#define COLOR_BUSH3 (0.55*vec3(0.1,0.35,0.09))
#define COLOR_BUSH4 (0.82*vec3(0.18,0.39,0.06))
#define COLOR_BUSH5 vec3(0.1,0.3,0.01) // rare color
#define COLOR_TREE1 (vec3(0.1,0.35,0.09)*0.55)
#define COLOR_TREE2 (vec3(0.1,0.45,0.08)*0.8)
#define COLOR_TREE_SURF vec3(0.15,0.4,0.04)
#define COLOR_GRASS vec3(0.1,0.35,0.09)
#define COLOR_GRASS2 vec3(0.35,0.39,0.06)
#define COLOR_MOWED_GRASS vec3(0.17,0.37,0.05)
#define COLOR_MOWED_GRASS2 (COLOR_MOWED_GRASS* 0.6)
#define COLOR_PINE (vec3(0.4,1.0,0.2)*0.2)
#define COLOR_PINE2 (vec3(0.5,1.0,0.0)*0.32)
#define COLOR_TRUNK (BROWN*0.5)
#define COLOR_ROOF1 (vec3(0.6,0.2,0.3)*0.7)
#define COLOR_ROOF2 (vec3(0.1,0.4,0.5)*0.7)
#define COLOR_BRICKWALL mix(vec3(0.52,0.33,0.22),vec3(0.9,0.9,0.7),0.35)
#define COLOR_HOUSE mix((vec3(1,1,1)*0.65),COLOR_BRICKWALL,0.1) // white is a bit too white so blend with brickwall color
#define COLOR_HOUSE_BOTTOM (COLOR_HOUSE*0.7)

struct CellPoint
{
    vec2 p;
    ivec2 _pi; // some cell index to take a hash from
    float pradius; // point radius, small points can be closer to cell edges
};

// meaning of id varies, if 32bits floats we should be able to have exact ints up to 2^24=16777216
struct DistanceId { float d; float id;  };

DistanceId MkDistanceId( float d, float id ) { DistanceId v; v.d = d; v.id = id; return v; }

DistanceId MkDistanceId_16( float d, float id, float id_fraction )
{
    return MkDistanceId( d, float( int( id ) | ( int( id_fraction * 65535.0 ) << 5 ) ) );
}

DistanceId MkDistanceId_5_5_5( float d, float id, vec3 v )
{
    return MkDistanceId( d, float( int( id )
                                   | ( int( v.x * 31.0 ) <<  5 )
                                   | ( int( v.y * 31.0 ) << 10 )
                                   | ( int( v.z * 31.0 ) << 15 ) ) );
}

vec2 DecodeId_16( DistanceId di )
{
    int i = int( di.id );
    return vec2( float( i&31 ), float( i >> 5 ) * ( 1.0 / 65535.0 ) );
}

float DecodeId( DistanceId di )
{
    int i = int( di.id );
    return float( i&31 );
}

vec4 DecodeId_5_5_5( DistanceId di )
{
    int i = int( di.id );
    return vec4( float( i&31 )
                 , float( ( i >>  5 ) & 31 ) * ( 1.0 / 31.0 )
                 , float( ( i >> 10 ) & 31 ) * ( 1.0 / 31.0 )
                 , float( ( i >> 15 ) & 31 ) * ( 1.0 / 31.0 ) );
}

#if 0 
// bogus build error on ? : in shadertoy
DistanceId opUdi( DistanceId a, DistanceId b ) { return a.d < b.d ? a : b; }
#else
DistanceId opUdi( DistanceId a, DistanceId b ) { if ( a.d < b.d ) return a; return b; }
#endif

// some rough scale for the roads pattern
// you need to tweak x spacing and y spacing of path slabs
// and the x and y scale of the main path function
#define LANEWIDTH 0.08

// 0: sin waves
//>1: sin waves with flat sections (default)
// 2: constant (straight lines, rectangular layout)
// 3: smooth noise
#define LANE_FUNC 1

vec2 PathNoise11sD( float x )
{
    x -= 0.5;
    float fr = fract( x );
    int x0 = int( x - fr );
    uint i = uint( x0 );
    float y0 = hash11u_lowbias32( i );
    float y1 = hash11u_lowbias32( i + 1u );
    float f = smoothstep_unchecked( fr );
    float fd = smoothstep_unchecked_d( fr );
    return vec2( mix( y0, y1, f ), ( y1 - y0 ) * fd );
}

// return lane's (y,y'), called a lot so make it cheap (this *does not* return world space y coord)
vec2 wavyLaneFuncAndDerivative( float x, float lane_index )
{
#if LANE_FUNC==0
    float a = mix( 0.5, 1.7, ( 1.0 + cos( lane_index ) ) * 0.5 ); // per lane constant
    return vec2( ( sin( a * x ) + 1.0 ) * 0.5, 0.5 * a * cos( a * x ) );
#elif LANE_FUNC==1
    // sin wave with soft straight sections so it's just all curvy
    float a = mix( 0.9, 2.0, ( 1.0 + cos( lane_index ) ) * 0.5 ); // per lane constant
    float z = ( 1.0 + sin( x * a ) ) * 0.5;
    return vec2( z * z, a * z * cos( a * x ) );
#elif LANE_FUNC==2
    return vec2( 0.5, 0 );
#elif LANE_FUNC==3
    return PathNoise11sD( x - lane_index * 80.0 );
#endif
}

#define PATH_SPACING_Y 1.0
// used by camera, don't forget to offset by lane_index (regular wavyLaneFuncAndDerivative() doesn't care!)
vec2 wavyLaneFuncAndDerivative_WORLD( float x, float lane_index )
{
    return wavyLaneFuncAndDerivative( x, lane_index ) + vec2( lane_index * PATH_SPACING_Y, 0 );
}

// return vector to closest point in .xy, derivative there in .zw
// use cheap when you don't care about accurate distance or closest point
vec4 sdWavyLane( vec2 p, float lane_index, bool cheap )
{
    vec4 ret;
    CLOSEST_POINT_TANGENT_TO_FUNCD_11_ARG_CHEAP( ret, p, wavyLaneFuncAndDerivative, lane_index, cheap )
    return ret;
}

struct SceneIn
{
    // constant during trace:
//  vec3 o;
    vec3 v;
    vec3 v_rcp; // for cell boundaries
    int trace_flags; // we can skip material related calculations during the traversal and enable only on shade
    // varying during trace:
    vec3 p;
    float t0; // warp start to hmax
    float t;
};

struct ClosestPath
{
    DistanceId patch_di; // patch id and distance (which is also the distance to paths)
    vec2 v2closest; // vector to closest patch boundary point, in world coords
    float curve_local_param; // the curve parameter, p.x in the local curve distance eval, so we can parametrixe space for making walls etc
                       // (since the path uses x and y directions we need to know which was used)
};

vec2 GetLocalCurvePoint( ClosestPath path ) { return vec2( path.curve_local_param, length( path.v2closest ) ); }

// spacing must be >= 1 for the band to work
ClosestPath EvalClosestPathSlab( vec2 p, float spacing, bool cheap )
{
    float id1 = floor( p.y / spacing );
    p.y = p.y - id1 * spacing;
    vec4 c1 = sdWavyLane( p, id1, cheap ); // should return in 0,spacing
    float d1 = cheap ? abs( c1.y ) : length( c1.xy );
    float s1 = sign( c1.y ); // this tells us if are above or below the curve (which adjacent curve to eval)

    float id2 = id1 - s1; // find the other 1d lane of the slab we are in: next or prev
    vec4 c2 = sdWavyLane( p + vec2( 0, s1 * spacing ), id2, cheap );
    float d2 = cheap ? abs( c2.y ) : length( c2.xy );
    float s2 = sign( c2.y );

    float m = s2 < 0.0 ? id1 : id1 - s1;

    ClosestPath eval;
    if ( d1 < d2 ) { eval.patch_di.d = d1; eval.v2closest = c1.xy; }
    else           { eval.patch_di.d = d2; eval.v2closest = c2.xy; }
    eval.patch_di.id = m;
    eval.curve_local_param = p.x;
    return eval;
}

// the lane curve is evaluated 4 times (twice per axis)
ClosestPath EvalClosestPath( vec2 p, bool cheap )
{
    // horizonal lanes
    // PATH_SPACING_Y must be >= 1 for the band to work
    ClosestPath ev1 = EvalClosestPathSlab( p, PATH_SPACING_Y, cheap );

    // vertical lanes
    // each horizontal stab can call a different set of vertical lanes which generates complex patterns
    vec2 p2 = perp( p ); // 90 degrees
    // important: spacing2 must be >= 1 for the band to work
    float spacing2 = ( 1.0 + 4.0 * ( sin( ev1.patch_di.id * 10.0 ) + 1.0 ) * 0.5 ); // spacinf of vertical lane can differ per horizontal stab which is key to complex branching patterns
    ClosestPath ev2 = EvalClosestPathSlab( p2, spacing2, cheap );
    ev2.v2closest = -perp( ev2.v2closest ); // put v2closest back to world coords (-90 degrees)

    ClosestPath eval = ev1;
    eval.patch_di.id *= ev2.patch_di.id; // get a unique id for this patch
    if ( ev2.patch_di.d < ev1.patch_di.d )
    {
        eval.patch_di.d = ev2.patch_di.d;
        eval.v2closest = ev2.v2closest;
        eval.curve_local_param = ev2.curve_local_param;
    }
    return eval;
}

// I hope you like magic constant
#define MAX_TERRAIN_HEIGHT 0.74
#define TALLEST_OBJECT_HEIGHT 0.645

float sin_bumps_array( vec2 p ) { return sin( p.x ) * sin( p.y ) + 1.0; }

// this function is called a lot so pick something simple
float BaseGroundHeight( vec2 p  )
{
    float h = sin_bumps_array( p.xy * ( 0.16 * PI ) ) * 0.5;
    return h * h * MAX_TERRAIN_HEIGHT; // sinbumps as is is too bumpy everywhere, we need also flatter areas, so square h
}

// image was blue point juniper... end result is noise vomit
float pine_tree_cross_section( float x, vec2 rh ) { x /= rh.y; return cubicbump( x, 3.0, -0.98 ) * rh.x; }
float sdRevolutionPine( vec3 p, float r, float h )
{
    vec2 p2d = vec2( ( p.z ), length( p.xy ) );
    if ( p2d.x >= h ) return length( p2d - vec2( h, 0 ) ); // don't forget to deal with endpoints...
    if ( p2d.x <= 0.0 ) return length( p2d - vec2( 0, 0 ) );
    float d; // note: we could use FAST_SIGNED_DISTANCE_TO_FUNCD_11_ARG but function is relatively cheap to eval so
    FAST_SIGNED_DISTANCE_TO_FUNC_11_ARG( d, p2d, pine_tree_cross_section, vec2( r, h ), 1e-3 );
    return d; // note: this is already signed
}

vec3 ApplyWind( vec3 pd, float aTime ) { return vec3( sin( aTime * 0.8 + pd.zx ) * 0.0003, 0 ); }

float ddPineSurfLayer( vec3 p )
{
    p *= 400.0;
    p.z *= 0.25;
    p.xy = rotate_with_angle( p.xy, p.z * 0.11 );
    return ( ( tri_sin( p.x ) + tri_sin( p.y ) + tri_sin( p.z ) ) + 3. ) * ( 1. / 6. ); // normalize to 0,1
}

float ddPineSurf( vec3 p )
{
    return ( ddPineSurfLayer( p ) + ddPineSurfLayer( vec3( rotate_with_angle( p.xy, 0.5 ), p.z ) * 0.5 ) )
           * ( 1. / ( 2.5 ) ); // normalize to 0,1
}

float sdDisplacePine( vec3 p, float d, float uheight )
{
    if ( !DISPLACE_PINE ) return d;
    float dd_amp = 0.018;
    if ( d > dd_amp ) return d;
    dd_amp *= ddPineSurf( p );
    return d + dd_amp;
}

#define TRACE_VIEW 1
#define TRACE_SHADE 2
#define TRACE_AO 4
#define TRACE_SHADOW 16
#define TRACE_HAS_DIRECTION 64 // note: it is faster to check flags than checking for v_rcp == 0 or whatnot (that might not been resolved to static)

bool IsShadeTrace( int trace_flags ) { return ( trace_flags & TRACE_SHADE ) != 0; }
bool IsShadowTrace( int trace_flags ) { return ( trace_flags & TRACE_SHADOW ) != 0; }
bool HasDirection( int trace_flags ) { return ( trace_flags & TRACE_HAS_DIRECTION ) != 0; }

DistanceId sdTreeLeaves( vec3 p, vec2 c, float cz, float trunk_height
                         , float leaves_volume_base_radius
                         , float leaves_volume_top_radius
                         , int trace_flags
                         , float color_rnd )
{
    float d = FLT_MAX;
    vec3 c0 = vec3( c, cz + trunk_height * 0.7 );
    vec3 c1 = vec3( c, cz + trunk_height );
    float d0 = length( p - c0 ) - leaves_volume_base_radius;
    float d1 = length( p - c1 ) - leaves_volume_top_radius;
    // soft blend between 2 sphere for the base leaves volumes
    float k = leaves_volume_base_radius * 0.4;
    d = opU_soft2_pol( d0, d1, k );
//  return MkDistanceId( d, MATID_TREE );

#define TREE_SDD 0.02 // upper bound for  amplitude of fine displacement on pines and trees

    float dd1 = 0.007;
    if ( d > dd1 + TREE_SDD ) return MkDistanceId( d, MATID_TREE ); // massive saving. only consider envelope displacement when we are close

    float d_envelope = d;
    bool blend = _1;
    float depth = 0.0;
    float uheight = 0.0;
    
    if ( _1 )
    {
        // make the leaves volume more bubbly by using 2 layers of *packed* 3d spheres regular tiling (non packed doesn't work well)
        // see @Shane's cellular tiling shaders/comments

        mat3 m3 =
            mat3_rotation_x( unit_vector2( radians( c.x * 200. ) ) ) *
            mat3_rotation_z( unit_vector2( radians( c.y * 200. ) ) );

        float dl1;

        {
            float c1 = leaves_volume_base_radius * 0.3;
            float da1 = packed_spheres_tiling3d( p * m3, c1, c1 * 2.5 ); // more spacing gives more clustering/bumpiness
            da1 += smoothstep( cz + trunk_height * 0.75, cz + trunk_height, p.z ) * 0.007;
            dl1 = blend
                ? opI_weld_quadric( d_envelope, da1, leaves_volume_base_radius * 0.3 )
                : opI( d_envelope, da1 );
        }

        float dl2;

        {
            // note: the second packed_spheres_tiling3d could just be global and evaluated 1 for the 3 trees
            float c1 = leaves_volume_base_radius * 0.4;
            float da1 = packed_spheres_tiling3d( m3 * p, c1, c1 * 2.5 );
            dl2 = blend
                ? opI_weld_quadric( d_envelope, da1, leaves_volume_base_radius * 0.25 )
                : opI( d_envelope, da1 );
        }

        float d_leaf_clusters = blend
            ? opU_weld_quadric( dl1, dl2, leaves_volume_base_radius * 0.1 )
            : opU( dl1, dl2 );

        d = blend
            ? opU_weld_quadric( d_leaf_clusters, d_envelope + 0.02, leaves_volume_base_radius * 0.19 )
            : d_leaf_clusters;

        depth = saturate( 1.0 / ( 1. + abs( d - d_envelope ) * 100. ) );
        
        float b = c0.z - leaves_volume_base_radius; // bottom most-ish
        float t = c1.z + leaves_volume_top_radius; // topmost-ish
        uheight = saturate( ( p.z - b )/( t - b ) ); // a 0-1 normalized height value for shading
    }

    if ( _1 )
    {
        // clip leaves volume's bottom with a wavy surface so it's not spherical things everywhere
        float clipsurf = ( ( c0.z - leaves_volume_base_radius * .6 )
                           + leaves_volume_base_radius * 0.1
                           * ( sin_bumps_array( p.xy * ( 13.0 * PI ) ) - 2.4 ) );

        d = blend
            ? opI_weld_quadric( d, -( p.z - clipsurf ), leaves_volume_base_radius * 0.15 )
            : opI( d, -( p.z - clipsurf ) );
    }

    return MkDistanceId_5_5_5( d, MATID_TREE, saturate( vec3( depth, uheight, color_rnd ) ) );
}

// retrieve terrain height, using full eval or taylor expansion
float CalcHeight( vec2 c, vec2 p, vec3 h_gradval, bool taylor_expansion_height )
{
    return taylor_expansion_height
           ? h_gradval.z + dot( h_gradval.xy, c - p ) // h(p)=h(c)+(c-h).grad(c): taylor expansion to skip height evaluations 
           : BaseGroundHeight( c );
}

DistanceId sdGridObj_TreeOrPine( DistanceId di
                                 , vec3 p
                                 , CellPoint cr
                                 , float radius_fraction
                                 , float patch_id
                                 , inout vec3 color
                                 , SceneIn scenein
                                 , vec3 h_gradval, bool taylor_expansion_height )
{
    vec2 c = cr.p;
    float r = radius_fraction * cr.pradius;
    float cz = CalcHeight( cr.p, p.xy, h_gradval, taylor_expansion_height );

    if ( _0 && ( ( length( p - vec3( c, cz ) ) - cr.pradius ) > di.d ) ) return di; // CULL
#if 0
    // doesn't save enough vs extra test cost
    if ( HasDirection( scenein.trace_flags ) )
    {
        vec2 n = perp( scenein.v.xy );
        vec2 o = scenein.o.xy;
        vec2 pp = project_on_plane( c, o, n );
        if ( lensqr( p.xy - pp ) > r * r ) return di;
    }
#endif
    vec4 hhh = hash42_( cr._pi * 123 );
    vec3 tree_base_point = vec3( c, cz );

    float pine_dice_roll = hhh.x;
    float pine_probability = 0.7;
    
    if ( _1 && ( pine_dice_roll > pine_probability) )
    {
        float pine_radius_scale = mix( 0.65, 0.75, hhh.w ) * r;
        float pine_height = mix( 0.42, 0.56, hhh.z * hhh.z );
        float uheight = saturate( ( p.z - cz ) / pine_height );
        DistanceId pine_tree = MkDistanceId_5_5_5( 
            sdDisplacePine( p - tree_base_point
                            , sdRevolutionPine( p - tree_base_point, pine_radius_scale, pine_height ), uheight )
            , MATID_PINE, vec3( 0.0, uheight, ( ( hhh.x - pine_probability ) * ( 1.0 / ( 1.0 - pine_probability ) ) ) ) );
        return opUdi( di, pine_tree );
    }

    hhh.x *= 1.0/pine_probability; // back into [0,1]
    float aa = mix( 0.018, 0.012, hhh.y ); // trunk
    float trunk_height = aa * ( 1.1 * 1.0 / 0.018 ) * mix( 0.2, 0.35, hhh.z * hhh.z );
    float trunk_radius = aa * 0.8;
    float leaves_volume_top_radius = mix( 0.5, 0.7, hhh.w ) * r;
    float leaves_volume_base_radius = mix( 1.1, 1.35, hhh.y ) * leaves_volume_top_radius;
    DistanceId leaves = sdTreeLeaves( p, c, cz, trunk_height, leaves_volume_base_radius, leaves_volume_top_radius, scenein.trace_flags, hhh.x );
    float trunk_uheight = saturate( ( p.z - cz ) / trunk_height );
    trunk_radius *= mix(0.8,1.2,pow2(1.0-saturate(trunk_uheight*3.5)));
    DistanceId trunk = MkDistanceId( sdCylinder( p - tree_base_point, vec2( trunk_radius, trunk_height ) ), MATID_TRUNK );
    if ( trunk.d < 0. ) leaves = trunk; // horrible hack to force trunk to be trunk inside leaves, as the leaves sdf has been hacked to death and union doesn't quite work anymore
    return opUdi( di, opUdi( trunk, leaves  ) );
}

// function used to make roof tiles
// a1 is the slope of curve going up (1.)
// a2 is the slope of curve going down (-2.)
// p is the period
float hard_waves( float x, float a1, float a2, float p ) { x = repeat( x, p ); return min( a1 * x, a2 * ( x - p ) ); }
// roof tiles height field (hf)
float hf_SurfaceRoofTiles( vec2 p ) { return 0.1 * hard_waves( p.y, 0.3, -1.1, 0.024 ) + 0.001 * ( 1.0 - pow2( 1.0 - abs( sin( p.x * 200. ) ) ) ); }

struct WindowOrDoorArg
{
    vec2 c; // cellsize
    vec2 g; // num cells
    float frame_width, border_height, border_depth, glass_depth, frame_depth;
};

void sdOneWindow( vec3 p, inout DistanceId eval, WindowOrDoorArg args )
{
    float d_in = eval.d;
    vec2 c = args.c;
    vec2 g = args.g - vec2( 1 );
    p.x += ( args.g.x * 0.5 - 0.5 ) * c.x; // center on x...
    vec2 pmin = -c.xy * 0.5;
    vec2 pmax = c.xy * ( vec2( 0.5 ) + g );
    // window glass and frame
    vec3 pr = p;
    vec2 i = floor( ( pr.xy - ( -c * 0.5 ) ) / c ); // c the cell size
    i = clamp( i, vec2( 0, 0 ), g );
    pr.xy -= i * c;
    float d_glass = sd_bounds_half_size( pr, vec3( c * 0.5 - vec2( args.frame_width ) * 0.5, args.glass_depth ) );
    eval.d = opS( eval.d, d_glass );
    // window general frame
    float d_frame = sd_bounds_range( p, vec3( pmin, -args.frame_depth ), vec3( pmax, args.frame_depth ) );
    eval.d = opS( eval.d, d_frame ); // make the whole window sink a bit
    // window border
    if ( _1 ) eval.d = opI( d_in - args.border_depth // clamp vs inflated version of source block we are decorating
                            , opU( eval.d
                                   , sd_bounds_range( p
                                                      , vec3( pmin.x, pmin.y - args.border_height, 0 )
                                                      , vec3( pmax.x, pmin.y, args.border_depth ) ) ) );
    if ( -d_glass == eval.d ) eval.id = MATID_WINDOW; // we used opS so we need -d_glass
}

void addWindow( inout DistanceId eval, vec3 p, float is_chimney_face, float half_wall_width )
{
    p.y += 0.016; // adjust windows height
    vec2 c = vec2( 0.07*mix(1.8,1.0,is_chimney_face), 0.07 ); // tile size
    vec2 window_size = vec2( 0.018, 0.02 ); // size of the object inside each tile, must be smaller than c
    float d_glass = FLT_MAX;
    vec2 i = floor( ( p.xy - ( -c.xy * 0.5 ) ) / c.xy ); // c the cell size
    float maxnum_cells = floor( (half_wall_width / c.x) - 0.5 ); // max num window that fit on this wall, assume p.x centered
    ivec2 imin = ivec2( -maxnum_cells, 1 );
    ivec2 imax = ivec2(  maxnum_cells, 1 );
    i = clamp( i, vec2(imin), vec2(imax) );
    p.xy -= i * c;
    WindowOrDoorArg args;
    args.c = window_size; // cellsize
    args.g = mix( vec2( 2, 2 ), vec2( 2, 2 ), is_chimney_face ); // window glass grid size
    float scl = 0.012;
    args.frame_width = 0.05 * scl;
    args.border_height = 0.3 * scl;
    args.border_depth = 0.2 * scl; // can't be bigger than d_house_bottom_inset
    args.glass_depth = 0.3 * scl;
    args.frame_depth = 0.1 * scl;
    sdOneWindow( p, eval, args );
}

// convert point p to y=0 face local point, a is a plane base3d/origin2d point
vec3 p2yface( vec3 p, vec2 orig ) { p.y = abs( p.y ); p.xy -= orig; p.xzy = p.xyz; p.x = -p.x; return p; }
// convert point p to y=0 face local point, a is a plane base3d/origin2d point
vec3 p2xface( vec3 p, vec2 orig ) { p.x = abs( p.x ); p.xy -= orig; return p.yzx; }

// note: tracing detail doesn't always work very well on heightfields so z proj/triplanar type of mapping isn't great for roof
// instead we do more tedious evals, building roof local points + 3d detail on that etc
DistanceId sdFarm( vec3 p, CellPoint cr,float patch_id,float r, float detail )
{
    vec4 hh = hash42_( cr._pi);
    bool has_chimney = hh.y > 0.4;
    bool _x2chimney = _1;
    vec2 hs1 = vec2( 0.25, 0.083 );
    vec2 hs2 = vec2( 0.083, 0.18+hh.z*0.02 );
    float h = 0.0996;
    float bottom_block_inset = 0.00498;
    float bottom_block_inset2 = bottom_block_inset*(has_chimney?0.0:1.0);
    float roof_thickness = 0.00166;
    float chimney_side_len = 0.018+hh.z*0.002; // chimney side length
    float chimney_height = h + hs1.y + chimney_side_len;
    float chimney_bottom = h + hs1.y - 0.04;
    float roof_tile_scl = 2.5; 
    float roof_tile_scl2 = 1.1; 
    bool half_hipped = hh.z>0.5;
    float bottom_inflate = 0.001;
    vec2 hs1_in = hs1 - vec2( bottom_block_inset2, bottom_block_inset );
    vec2 hs2_in = hs2 - vec2( bottom_block_inset );
    // symmetric window plane
    float block1_chimney_wall_plane_x = hs1_in.x;
    float block1_wall_plane_y = hs1_in.y;
    float block2_wall_plane_y = hs2_in.y;
    vec3 ps = p; // store signed p
    vec3 pay = p; pay.y = abs( pay.y ); // symmetric around y
    vec3 pax = p; pax.x = abs( pax.x ); // symmetric around x
    float d_block1_bottom = FLT_MAX;
    float d_block1_roof = FLT_MAX;
    float d_block1_bottom2 = FLT_MAX;
    {
        // --- gable roof, 2 planes (v-shaped)
        bounds2 block1 = mkbounds_unchecked( -hs1, hs1 );
        float d_block1_footprint = sd_bounds_range( p.xy, block1.pmin.xy, block1.pmax.xy );
        vec3 roof1_top_point = vec3( hs1.x, 0, h + hs1.y );
        // p.yz is the gable roof cross section space point
        vec3 roof_plane_local_p = vec3( -dot( pay.yz - roof1_top_point.yz, perp( V45 ) ), p.x, dot( pay.yz - roof1_top_point.yz, V45 ) );
        d_block1_bottom = opI( roof_plane_local_p.z, d_block1_footprint );
        d_block1_bottom = opI( d_block1_bottom, pay.y -hs1_in.y );
        d_block1_bottom = opI( d_block1_bottom, pax.x - block1_chimney_wall_plane_x );
        d_block1_bottom = opI( d_block1_bottom, roof_plane_local_p.z + 0.002 );
        // add tile detail to gabble roof, hacky mess to be sorted
        d_block1_roof = roof_plane_local_p.z - detail*hf_SurfaceRoofTiles( roof_plane_local_p.yx * roof_tile_scl )* roof_tile_scl2;
        d_block1_roof = opS( d_block1_roof, roof_plane_local_p.z + 0.002 );
        d_block1_roof = opI( d_block1_roof, d_block1_footprint );
        d_block1_bottom2 = opI(d_block1_footprint-bottom_inflate,p.z);
    }
    float d_chimney = FLT_MAX;
    if ( has_chimney )
    {
        vec2 chimney_c = vec2( hs1_in.x, 0 );
        bounds2 chimney_footprint_b = mkbounds_unchecked( chimney_c - vec2( chimney_side_len ), chimney_c + vec2( 0, chimney_side_len * 0.5 ) );
        float d_chimney_footprint = sd_bounds_range( _x2chimney ? abs( p.xy ) : p.xy, chimney_footprint_b.pmin.xy, chimney_footprint_b.pmax.xy );
        d_chimney = opI( chimney_bottom - p.z, opI( d_chimney_footprint, p.z -chimney_height ) );
        float d_chimney_hole = opI( chimney_bottom - p.z, opI( d_chimney_footprint, p.z - chimney_height * 1.5 ) ) + 0.002;
        d_chimney = opS( d_chimney, d_chimney_hole );
    }
    float d_block2_roof = FLT_MAX;
    float d_block2_bottom = FLT_MAX;
    float d_block2_bottom2 = FLT_MAX;
    if ( hh.x > 0.4 )
    {
        // --- hipped roof
        bounds2 block2 = mkbounds_unchecked( -hs2, hs2 );
        float d_block2_footprint = sd_bounds_range( p.xy, block2.pmin.xy, block2.pmax.xy );
        //return d_block1_roof;
        d_block2_bottom = opI( d_block2_footprint + bottom_block_inset, ( p.z - h ) ); // block2 is inset equally on x and y 
        d_block2_bottom2 = opI(d_block2_footprint-bottom_inflate,p.z);
        
        //return d_block2_bottom;
        vec3 roof2_corner_point = vec3( hs2, h );
        vec3 roof2a_plane_local_p = vec3( p.y, -dot( pax.xz - roof2_corner_point.xz, perp( V45 ) ),dot( pax.xz - roof2_corner_point.xz, V45 ) );
        float roof2detail_a = roof2a_plane_local_p.z - detail*hf_SurfaceRoofTiles( roof2a_plane_local_p.xy * roof_tile_scl ) * roof_tile_scl2;
        // like the s1 one, exactly. just offset, can we factorize?
        vec3 roof2b_plane_local_p = vec3( p.x, -dot( pay.yz - roof2_corner_point.yz, perp( V45 ) ),dot( pay.yz - roof2_corner_point.yz, V45 ) );
        float roof2detail_b = roof2b_plane_local_p.z - detail*hf_SurfaceRoofTiles( roof2b_plane_local_p.xy * roof_tile_scl ) * roof_tile_scl2;
        d_block2_roof = opI( roof2b_plane_local_p.z, roof2a_plane_local_p.z );
        d_block2_roof = opI( roof2detail_a, roof2detail_b );

        if ( half_hipped )
        {
            d_block2_roof = opI( d_block2_roof, p.y );
            d_block2_bottom = opI( d_block2_bottom, p.y );
            d_block2_bottom2 = opI( d_block2_bottom2, p.y );
        }
    }
    DistanceId bb1 = MkDistanceId( d_block1_bottom, MATID_HOUSE );
    DistanceId bb2 = MkDistanceId( d_block2_bottom, MATID_HOUSE );
    if ( FARM_WINDOWS )
    {
        addWindow( bb1, p2xface( p, vec2( hs1_in.x, 0.0 ) ), 1., hs1_in.y );
        addWindow( bb1, p2yface( p, vec2( 0, hs1_in.y ) ), 0., hs1_in.x );
        addWindow( bb2, p2yface( p, vec2( 0, hs2_in.y ) ), 0., hs2_in.x );
    }
    DistanceId roof_eval = MkDistanceId( opI( h - p.z, opU( d_block1_roof, d_block2_roof ) ), MATID_ROOF );
    DistanceId bottom_eval = MkDistanceId( d_chimney, MATID_HOUSE );
    bottom_eval = opUdi( bottom_eval, bb1 );
    bottom_eval = opUdi( bottom_eval, bb2 );
    DistanceId bottom2_eval = MkDistanceId( opU( d_block1_bottom2, d_block2_bottom2 ), MATID_HOUSE_BOTTOM );
    return opUdi( opUdi( roof_eval, bottom_eval ), bottom2_eval );
}

DistanceId sdGridObj_Farm( DistanceId di
                           , vec3 p
                           , CellPoint cr
                           , float radius_fraction
                           , float patch_id
                           , inout vec3 color
                           , SceneIn scenein
                           , vec3 h_gradval, bool taylor_expansion_height )
{
    vec2 c = cr.p;
    float r = radius_fraction * cr.pradius;
    float cz = CalcHeight( cr.p, p.xy, h_gradval, taylor_expansion_height );
    
    if ( _0 && ( ( length( p - vec3( c, cz ) ) - cr.pradius ) > di.d ) ) return di; // CULL
#if 0
    if ( HasDirection( scenein.trace_flags ) 
         && ( sphere_trace( mkray( scenein.p, scenein.v ), r, vec3( c, cz ) ).x == FLT_MAX ) )
    {
        return di;
    }
#endif
    // orient farm along terrain gradient to reduce sinking cases
    vec2 e = vec2( 1e-3, 0 );
    vec2 h_gradval_at_c = ( vec2( BaseGroundHeight( c.xy + e.xy ),
                                  BaseGroundHeight( c.xy + e.yx ) ) - vec2( cz ) ) / e.x; // be careful to not divide by e since it has zero in .y
    
    float grad_len = length( h_gradval_at_c );

    #define FARMS_GRAD_LIMIT   _1 // don't place farms when slope is too strong
    #define FARMS_GRAD_ELEVATE _1 // elevate house so they don't sink into the ground, using the gradient
    #define FARMS_GRAD_ALIGN   _1 // align farm with terrain gradient instead of random rotation

    if ( FARMS_GRAD_LIMIT && ( grad_len > 0.17 ) ) return di;
    if ( FARMS_GRAD_ELEVATE ) cz += r * grad_len * 0.65;

    // move to local coords
    p.xy -= c;
    p.z -= cz;
    
    p.xy = rotate_with_unit_vector( p.xy, FARMS_GRAD_ALIGN && ( grad_len > 0.01 )
                                    ? normalize( h_gradval_at_c ) 
                                    : unit_vector2( hash1u2_4tXyWN( uvec2(cr._pi) ) * 2.0 * PI ) ); // random rotation

    if ( _0 ) return MkDistanceId( sd_sphere( p, vec3( 0 ), r ), MATID_HOUSE ); // try make the building fit inside the sphere

    DistanceId eval = sdFarm( p, cr,patch_id, r, 1.0 );
    
    if ( IsShadeTrace( scenein.trace_flags ) && ( DecodeId( eval ) == MATID_ROOF ) )
    {
        eval = MkDistanceId_16( eval.d, MATID_ROOF, hash11( patch_id ) ); // same roof color for all houses in patch
    }

    return eval;
}

float sdBrickWall( vec3 p, ClosestPath path, float h )
{
    float wall_start_dist = LANEWIDTH * 0.5 + 0.005;
    float wall_thickness = 0.018;
    float wall_height = 0.05;
    float wall_radius = 0.008;
    float d = FLT_MAX;
    for ( float k = 0.0; k < 2.0; k += 1.0 )
    {
        vec3 pl = p; // p local
        pl.xy = GetLocalCurvePoint( path );
        pl.y -= wall_start_dist + wall_thickness * 0.5;
        pl.z -= h;
        vec3 s = vec3( 0.01, 0.006, 0.006 );
        float spacing = 0.0015;
        vec3 c;
        c.xz = vec2( s.x + spacing, 2.0 * s.z + spacing );
        c.y = s.y + spacing;
        vec2 offset = -0.5 * c.xz;
        float o = 0.5 * k;
        offset.xy += o * c.xz;
        vec2 i;
        i = floor( ( pl.xz - offset ) / c.xz );
        i.y = min( i.y, 2.0 );
        pl.xz -= i * c.xz;
        pl.xz -= o * c.xz;
        float r = 0.002;
        d = opU( d, sd_bounds_range_round( pl, -s * 0.5, s * 0.5, r ) );
    }
    if ( _1 ) d -= 0.003 * sfbm2_13( p * 80.0 );
    return d;
}

float sdGrass( vec3 p, float h ) { return p.z - h; }

// scene eval output
struct SceneOut
{
    DistanceId object_di;
    float d_ghost; // can we optimize that and have less of those?
    float base_height; // base ground height
    ClosestPath path;
    ClosestPath test2d; // for 2d view mode
    vec3 color; // special color case for bushes (when id_fraction is not enough)
#ifdef TERRAIN_WARP
    // this cuts bush evals massively
    float terrain_warp; // todo: move out of this struct, this is not part of returned information
#endif
};

void SceneOutInit( inout SceneOut eval ) { eval.d_ghost = FLT_MAX; }

vec3 get_bush_palette( vec2 uv )
{
    return mix( mix( mix( COLOR_BUSH1, COLOR_BUSH2, uv.x ), mix( COLOR_BUSH3, COLOR_BUSH4, uv.x ), uv.y )
                , COLOR_BUSH5, smoothband( uv.x, 0.49, 0.51, 0.01 ) );
}

bool is_white_cell( vec2 p_index ) { return ( int( p_index.x + p_index.y ) & 1 ) == 1; }

void consider_close_point_hi( inout float d, vec2 index, float r, inout vec4 color, vec3 p, float cellsize
                              , vec3 h_gradval, bool taylor_expansion_height, int trace_flags )
{
    vec3 c;
    vec3 h = hash32( index );
    c.xy = ( index + h.xy ) * cellsize;
    c.z = CalcHeight( c.xy, p.xy, h_gradval, taylor_expansion_height ) + r * mix( -0.8, 1.3, h.z );
    float di = length( c - p ) - r;
    d = min( d, di );

    if ( IsShadeTrace( trace_flags ) )
    {
        vec2 ch = hash22( index );
        // this is called only once in shade, knock yourself out
        float w = max( 1. - smoothstep( -r * 0.1, r * 0.25, di ), 1e-3 );
        color.xyz += get_bush_palette( ch ) * w;
        color.a += w;
    }
}

struct CloseGridPointArgs { float cell_size, max_radius, radius_disparity; };

struct CloseGridPointArgsWithBand
{
    CloseGridPointArgs args0;
    float band_start, band_end;
    bool taylor_expansion_height;
};

// 45 means we consider 4 or 5 neighbour (instead of 3x3) depending on whether we are on a white or a black cell (not perfect but covers lots of cases)
// instead of giving the closest point this version does a little bit of extra calculation or each candidate for color blending on bushes
float GetCloseGridPoints45( inout vec3 a_color, vec3 p, float cellsize, vec3 h_gradval, CloseGridPointArgsWithBand args, int trace_flags, int aFrame )
{
    float r = args.args0.max_radius * 0.8;
    vec2 p_index = floor( p.xy * ( 1.0 / cellsize ) );

    float d = FLT_MAX;
    vec4 color = vec4( 0.0 );
#if 0
    // the unrolled code path is faster on my current view (70ms->68ms) but shader compilation prefers the loop (-1s)
    if ( is_white_cell( p_index ) )
    {
        //white cell
        consider_close_point_hi( d, p_index + vec2( -1,  0 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
        consider_close_point_hi( d, p_index + vec2(  1,  0 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
        consider_close_point_hi( d, p_index + vec2(  0, -1 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
        consider_close_point_hi( d, p_index + vec2(  0,  1 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
    }
    else
    {
        //black cell
        consider_close_point_hi( d, p_index + vec2( -1,  1 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
        consider_close_point_hi( d, p_index + vec2(  1,  1 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
        consider_close_point_hi( d, p_index + vec2( -1, -1 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
        consider_close_point_hi( d, p_index + vec2(  1, -1 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
        consider_close_point_hi( d, p_index + vec2(  0,  0 ), r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
    }
#else
    vec2 offset = vec2( 0, 1 ); // white cells check 4 canonical axis neigbours
    if ( !is_white_cell( p_index ) )
    {
        consider_close_point_hi( d, p_index, r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags ); // this cell, only checked if black
        offset = vec2( 1, 1 ); // black cells check the 4 diagonal neighbours
    }

    for ( int i = 0 FORCE_LOOP; i < 4; ++i )
    {
        consider_close_point_hi( d, p_index + offset, r, color, p, cellsize, h_gradval, args.taylor_expansion_height, trace_flags );
        offset = perp( -offset ); // go to next neighbour by rotating +90
    }
#endif
    if ( IsShadeTrace( trace_flags ) ) // skipping this test breaks the trees...
        a_color = color.xyz / color.a;

    return d;
}

#define bush_max_radius 0.04
// for bushes we do something symbolic looking, with a slight Hiroshi Nagai vibe 
void AddContributionBushes( inout SceneOut eval, SceneIn scenein
                            , vec3 h_gradval, float groundz, vec3 patch_id_hash
                            , bool is_forest_patch, float aTime, int aFrame )
{
    vec3 p = scenein.p;
    bool has_bushes = patch_id_hash.x > 0.2;
    bool has_cut_bushes = SQUARE_BUSH_TEST && ( !is_forest_patch && patch_id_hash.x > 0.62 );

    float bush_cell_size = 0.095;

    float bush_d_min = p.z - ( groundz + bush_max_radius*1.9 ); // upper bound for distance to bushes
                                                                  // multiply by 2 else lots of bushes get cut... 

    // note: whether we use > or <= has resulted in big difference in the past sometimes
    if ( !has_bushes
         || ( bush_d_min > eval.object_di.d )
#ifdef TERRAIN_WARP
         || (
              ( eval.terrain_warp != 0.0 ) &&
              ( bush_d_min > 0.0 ) &&
              ( bush_d_min < eval.terrain_warp ) ) // we know we can roughly traverse by eval.terrain_warp before hitting next bush
#endif
        ) return;

    float d = FLT_MAX;
    float id_fraction = 0.0;
    float freq = 1.0;

    if ( has_cut_bushes )
    {
        freq = DISPLACE_BUSH_FREQ;
        
        // patch section 2d point
        vec2 p_patch_section = vec2( eval.path.patch_di.d, p.z - eval.base_height );
        float d_square_bush = sd_bounds_range( p_patch_section, vec2( LANEWIDTH - 0.015, 0 ), vec2( LANEWIDTH + 0.02, 0.05 ) );
        d = d_square_bush; // we distort so...

        eval.color = COLOR_BUSH3;
    }
    else
    {
        freq = DISPLACE_BUSH_FREQ;

        // this is a patch with bushes on the side
        CloseGridPointArgsWithBand args;
        args.args0.cell_size = bush_cell_size;
        args.args0.max_radius = bush_max_radius;
        args.args0.radius_disparity = 0.4;
        args.band_start = LANEWIDTH * 0.5 + args.args0.max_radius * 0.5; // take a fraction of the radius so that some of the bushes overlap a bit with the path
        args.band_end = args.band_start + 0.18 + pow2( patch_id_hash.z ) * 0.3;
        args.taylor_expansion_height = TAYLOR_HEIGHT_BUSH;

        d = GetCloseGridPoints45( eval.color, p, args.args0.cell_size * 0.5, h_gradval, args, scenein.trace_flags, aFrame );

        if ( SOFT_BLEND_BUSH ) d = opU_weld_quadric( p.z - eval.base_height, d, 0.027 );

        // clip bush vs path/walls (leak through walls is a happy accident)
        if ( _1 ) d = opS_soft2( d, abs( eval.path.patch_di.d ) - LANEWIDTH * 0.5, 100.0 );

        if ( _1 )
        {
            // fade bush inside patch
            float bf = ( sfbm1_12( p.xy * 18.0 ) - 0.5 ) * 0.33; // distort fade boundary
            d = opS_soft2( d, args.band_end * ( 1. + bf ) - eval.path.patch_di.d, 30.0 );
        }
    }

    if ( DISPLACE_BUSH )
    {
        float dd = 0.0045; // need more displacement to see shadows...

        vec3 pd = p;
        if ( WIND_BUSH ) pd += ApplyWind( pd, aTime );
        
         // the test is < ..*3 because we INFLATE
        if ( d < dd * 3.0 ) d -= sfbm2_13( pd * vec3( 80.0, 80.0, 100 ) * freq ) * dd;
    }

    eval.object_di = opUdi( eval.object_di, MkDistanceId_16( d, MATID_BUSH, id_fraction ) );
}

// return closest cell point with a radius, no neighbour, used by farms and trees
// max_radius must be less than cell_size*0.5
// radius_disparity percentage in 0,1
// grid_offset in 0,1, conceptually...
void GetClosestGridPoint( inout CellPoint point, vec2 p, CloseGridPointArgs args0, float grid_offset, float hoffset )
{
    vec2 pi = floor( ( p - grid_offset ) / args0.cell_size );
    point._pi = ivec2( pi + vec2( hoffset ) );
    vec4 ph = hash42_( point._pi );
//  ph.xy = vec2( 0.5 ); // debug
    point.pradius = args0.max_radius * ( 1.0 - args0.radius_disparity * ph.z );
    vec2 a = grid_offset + pi * args0.cell_size;
    vec2 b = a + vec2( args0.cell_size );
    point.p = mix( a + vec2( point.pradius ),
                   b - vec2( point.pradius ), ph.xy ); // important: +offset to put back in same space as p
}

// used by farms and trees
bool GetClosestGridPointWithPathBand_x1( inout CellPoint point
                                         , vec2 p
                                         , vec3 h_gradval
                                         , CloseGridPointArgsWithBand args
                                         , float grid_offset, float hoffset )
{
    GetClosestGridPoint( point, p, args.args0, grid_offset, hoffset );

    // we want to know if this point is within a band of the country patch we are currently in
    float distance_to_patch_border = abs( EvalClosestPath( point.p, true ).patch_di.d );
    return ( distance_to_patch_border > args.band_start )
        && ( distance_to_patch_border < args.band_end );
}

void SetSceneInDirection( inout SceneIn scenein, vec3 o, vec3 v, int trace_flags )
{
    scenein.v = v;
    scenein.v_rcp = vec3( 1.0 ) / v;
//  scenein.o = o;
    scenein.trace_flags = trace_flags | TRACE_HAS_DIRECTION;
    scenein.t0 = 0.0;
}

void SetSceneInDirectionless( inout SceneIn scenein, int trace_flags )
{
    scenein.v = vec3( 0.0 );
    scenein.v_rcp = vec3( 0.0 );
//  scenein.o = vec3( 0.0 );
    scenein.trace_flags = trace_flags;
    scenein.t0 = 0.0;
}

float ClampRayAgainstCurrentGridCell( vec3 p, vec3 v_rcp, vec3 cell_size, float cell_inflate_epsilon )
{
    // we know the direction therefore we only have to test one side of each axis
    vec3 s = sign( v_rcp );
    vec3 amin = floor( p / cell_size ) * cell_size;
    amin -= vec3( cell_inflate_epsilon ); // instead of adding that to d_ghost we clamp to cell_size + cell_inflate_epsilon
    vec3 a = amin + ( ( s + 1.0 ) * 0.5 ) * ( cell_size + 2.0 * vec3( cell_inflate_epsilon ) );
    vec3 t = ( a - p ) * v_rcp; // ray vs all closest box planes
//  t += FLT_MAX * ( 1.0 - abs( s ) ); // when sign is zero, push next hit at t=+infinite, v_rcp has been set to zero in that case
    return min( min( t.x, t.y ), t.z ); // should be > 0 by construction
}

bool CanGhostStep( SceneIn scenein, float maxdist_from_t0, float maxdist_abs )
{
    return HasDirection( scenein.trace_flags )
        && ( scenein.t < maxdist_abs ) // GHOST_MAX_DIST_ABS
        && ( ( scenein.t - scenein.t0 ) < maxdist_from_t0 ); // GHOST_MAX_DIST_FROM_T0
}

void GhostGridSteps( inout SceneOut eval, SceneIn scenein, float cell_size, float cell_inflate_epsilon )
{
    float dm = ClampRayAgainstCurrentGridCell( scenein.p, scenein.v_rcp
                                               , vec3( cell_size, cell_size, 20.0 ), cell_inflate_epsilon );
    // note that this doesn't deal with diagonals which might cross another cell
    eval.d_ghost = min( eval.d_ghost, max( dm, GHOST_EPS ) );
}

// used by farms and trees
#define SD_OBJECT_CONTRIB( _di_, _object_sd_func_, _grid_offset_, _hoffset_ ) {\
    CellPoint _cp_; \
    if ( GetClosestGridPointWithPathBand_x1( _cp_, p.xy, h_gradval, args, _grid_offset_, _hoffset_ ) )\
        _di_.object_di = _object_sd_func_( _di_.object_di, p, _cp_, radius_fraction, _di_.path.patch_di.id, eval.color, scenein, h_gradval, args.taylor_expansion_height );\
}

void AddContributionTrees( inout SceneOut eval, SceneIn scenein
                           , vec3 h_gradval, float groundz, vec3 patch_id_hash, int aFrame, float aTime )
{
    vec3 p = scenein.p;
    float tree_max_radius = 0.15;

    CloseGridPointArgsWithBand args;
    args.args0.cell_size = 0.4;
    args.args0.max_radius = tree_max_radius;
    args.args0.radius_disparity = 0.1;
    args.band_start = args.args0.max_radius;
    // some patches are filled with trees, some other patches only have trees on the border
#if 1
    args.band_end = patch_id_hash.x > 0.9 ? FLT_MAX : args.args0.max_radius * 3.0;
#else
    args.band_end = FLT_MAX;
#endif
    args.taylor_expansion_height = TAYLOR_HEIGHT_TREE;

    float grid_offset_step = 0.25; // this is wrong actually but creates a certain sparsity that is welcome
    vec3 grid_offsets = vec3( 0., 1.0, 2.0 ) * grid_offset_step * args.args0.cell_size;
    float radius_fraction = 1.0;
#if 0
    SD_OBJECT_CONTRIB( eval, sdGridObj_TreeOrPine, grid_offsets.x, 000.0 );
    SD_OBJECT_CONTRIB( eval, sdGridObj_TreeOrPine, grid_offsets.y, 100.0 );
    SD_OBJECT_CONTRIB( eval, sdGridObj_TreeOrPine, grid_offsets.z, 200.0 );
#else
    // this might be a bit faster (92->71!), also shadertoy compile time
    for ( float f = 0.0 FORCE_LOOPF; f < 3.0; f += 1.0 )
    {
        SD_OBJECT_CONTRIB( eval, sdGridObj_TreeOrPine, grid_offset_step * args.args0.cell_size * f, 100.0 * f );
    }
#endif

    float mat_id = DecodeId( eval.object_di );
    // displace all the trees at once...
    if ( DISPLACE_TREE && ( ( mat_id == MATID_TREE ) || ( mat_id == MATID_PINE ) ) )
    {
        vec3 pd = p;
        if ( WIND_TREE_AND_PINES ) pd += ApplyWind( pd, aTime );

        float is_pine = ( mat_id == MATID_PINE ) ? 1. : 0.;
        
        pd.z *= mix(1.,0.75,is_pine);
        
        eval.object_di.d += sfbm2_13_leaf( pd * 80.0 * 2.2 ) * TREE_SDD * mix( 1., 0.4, is_pine ) * 0.8;
    }

    if ( GHOST_STEPS_TREE && CanGhostStep( scenein, GHOST_MAX_DIST_FROM_T0, GHOST_MAX_DIST_ABS ) )
    {
        // note: offset zero for all + smallest cell size fraction multiple should yield same result
        float cell_inflate_epsilon = 0.004; // tree
        GhostGridSteps( eval, scenein, grid_offset_step * args.args0.cell_size, cell_inflate_epsilon );
    }
}

void AddContributionFarms( inout SceneOut eval, SceneIn scenein
                           , vec3 h_gradval, float groundz, vec3 patch_id_hash
                           , float closest_path_middle_point_height )
{
    vec3 p = scenein.p;
    bool is_farm_patch = patch_id_hash.x > 0.2;

    if ( !is_farm_patch ) return;

    if ( FARM )
    {
        // farms are sparse so use 1 cell
        CloseGridPointArgsWithBand args;
        args.args0.cell_size = 1.2;
        args.args0.max_radius = 0.35;
        args.args0.radius_disparity = 0.3;
        args.band_start = args.args0.max_radius * 1.5;
        args.band_end = FLT_MAX;
        args.taylor_expansion_height = TAYLOR_HEIGHT_FARM;
        float radius_fraction = 0.8;
        SD_OBJECT_CONTRIB( eval, sdGridObj_Farm, 0.0, 0.0 );
        if ( GHOST_STEPS_FARM && CanGhostStep( scenein, GHOST_MAX_DIST_FROM_T0, GHOST_MAX_DIST_ABS ) )
        {
            float cell_inflate_epsilon = 0.004;
            GhostGridSteps( eval, scenein, args.args0.cell_size, cell_inflate_epsilon );
        }
    }

    if ( BRICKWALL && patch_id_hash.y > 0.7 ) // note: not all farm patch have a visible house on them...
    {
        DistanceId walls = MkDistanceId( sdBrickWall( p, eval.path, eval.base_height ), MATID_BRICKWALL );
        eval.object_di = opUdi( eval.object_di, walls );
    }
}

void AddContributionGrass( inout SceneOut eval, vec3 p, float groundz, vec3 patch_id_hash )
{
    float grass_height = ( groundz - 0.01 ) // make sure grass doesn't cover the path ground
        + smoothstep( -0.02, 0.04, abs( eval.path.patch_di.d ) - LANEWIDTH * 0.5 ) * 0.0195;
    DistanceId grass = MkDistanceId_16( sdGrass( p, grass_height ), MATID_GRASS, patch_id_hash.z ); // each patch has slightly different colors
    eval.object_di = opUdi( eval.object_di, grass );
}

void AddContributionPath( inout SceneOut eval, SceneIn scenein, float groundz )
{
    vec3 p = scenein.p;
    float path_d_min = p.z - groundz;

    if ( !( path_d_min < eval.object_di.d ) ) return; // weird compiler horror happened depending how we write the if branch here

    // path is the most occluded thing so do it last
    DistanceId path_di = MkDistanceId_16( path_d_min // lower bound for bushes
                                       , MATID_GROUND, ( 1.0 - smoothstep( 0.0, 0.05, eval.path.patch_di.d ) ) );
    if ( DISPLACE_PATH )
    {
        // this displacement is expensive, cull as much as we can, maybe we could just do that in shade
        float path_blend = 1.0 - smoothstep( 0.001, 0.008, eval.path.patch_di.d - LANEWIDTH * 0.5 );
        float distance_blend = 1.0 - smoothstep( 6.0, 7.0, scenein.t );
        float fade = path_blend * distance_blend;
        if ( fade > 0.0 ) // that cuts a bit
        {
            // so we can scale along road direction
            path_di.d += fade * sfbm2_12( GetLocalCurvePoint( eval.path ) * vec2( 1, 2 ) * 80.0 ) * DISPLACE_PATH_AMPL;
        }
    }

    eval.object_di = opUdi( eval.object_di, path_di );
}

SceneOut evalScene( SceneIn scenein, int aFrame, float aTime)
{
    vec3 p = scenein.p;

    SceneOut eval;
    SceneOutInit( eval );
    
    float groundz = BaseGroundHeight( p.xy );
    eval.base_height = groundz;
    eval.path = EvalClosestPath( p.xy, false );
    vec3 h_gradval = vec3( 0.0, 0.0, groundz );

#ifdef TERRAIN_WARP
    vec3 ground_normal;
#endif

    if (
#ifdef TERRAIN_WARP
         _1 ||
#endif
         // any of those need the height gradient at p
         TAYLOR_HEIGHT_BUSH ||
         TAYLOR_HEIGHT_TREE ||
         TAYLOR_HEIGHT_FARM )
    {
        vec2 e = vec2( 1e-3, 0 );
        float hx = BaseGroundHeight( p.xy + e.xy );
        float hy = BaseGroundHeight( p.xy + e.yx );
        h_gradval.xy = vec2( hx - eval.base_height,
                             hy - eval.base_height ) / e.x; // be careful to not divide by e since it has zero in .y
#ifdef TERRAIN_WARP
        vec3 px = vec3( p.xy + e.xy, hx );
        vec3 py = vec3( p.xy + e.yx, hy );
        vec3 pc = vec3( p.xy, eval.base_height );
        ground_normal = normalize( cross( px - pc, py - pc ) );
#endif
    }

#ifdef TERRAIN_WARP
    eval.terrain_warp = 0.0;
    if ( HasDirection( scenein.trace_flags ) )
    {
        float large_optimistic_step = 3.0;
        vec3 base = vec3( p.xy, eval.base_height + bush_max_radius*2.5 );
        Ray warp_ray;
        warp_ray.o = scenein.p;
        warp_ray.d = scenein.v;
        float t2 = plane_trace( warp_ray, base, ground_normal, 1e-3 );
        if ( t2 > 0.0 )
        {
//          if ( dot( ground_normal, scenein.v ) < 0 ) // if terrain is convex at ground_normal in trace direction?
                eval.terrain_warp = min( t2, large_optimistic_step );
        }
    }
#endif

    // eval.path.patch_di.id is the patch id
    // eval.path.patch_di.d is the distance to closest path
    //
    // the path we walk on is flat so we need the height of center of the road
    vec2 closest_patch_border_point2 = p.xy + eval.path.v2closest; // center of road
    float closest_path_middle_point_height = BaseGroundHeight( closest_patch_border_point2 ); // fences use

    eval.test2d = eval.path;

    // upper bound for distance to ground, take into account displacement that might dig a little on paths
    float d_ground_max = p.z - ( eval.base_height - DISPLACE_PATH_AMPL * 2.0 );
#if 0
    // bogus build error on ? : in shadertoy
    eval.object_di = GROUND_OCCLUSION // occlusion helps
        ? MkDistanceId( d_ground_max, MATID_GROUND )
        : MkDistanceId( FLT_MAX, MATID_NONE ); // enable PATH to get a ground
#else
    if ( GROUND_OCCLUSION ) eval.object_di = MkDistanceId( d_ground_max, MATID_GROUND ); // occlusion helps
    else eval.object_di = MkDistanceId( FLT_MAX, MATID_NONE ); // enable PATH to get a ground
#endif
    // hit the patch boundary tangent plane, this has 2 properties we want:
    //  1- the closer we are to the boundary the more this approximates the hit point, sort of
    //  2- if ray leaches the patch boundary, the hit point will be far away
    //      -> that second one improves significantly the horrible artifact where we run out of points...
    //  also ignore far away hits for perfs   
    
    // todo: we don't need ghost steps if we are far above the ground! we can save a bit

    if ( GHOST_STEPS_PATCH && CanGhostStep( scenein, GHOST_MAX_DIST_FROM_T0, GHOST_MAX_DIST_ABS ) )
    {
        vec3 base = vec3( p.xy + eval.path.v2closest, p.z );
        vec3 normal = normalize( -vec3( eval.path.v2closest, 0 ) );
        float t2 = plane_trace( p, scenein.v, base, normal, 1e-3 );
        if ( t2 > 0.0 ) eval.d_ghost = min( eval.d_ghost, max( t2, GHOST_EPS ) );
    }

//  vec3 patch_id_hash = hash31( eval.path.patch_di.id );
    vec3 patch_id_hash = hash42_( ivec2( int( eval.path.patch_di.id ) ) ).xyz; // paranoid use of ints for important structural elements

    bool is_forest_patch = patch_id_hash.x > 0.65;

    if ( is_forest_patch )
    {
        if ( TREE ) AddContributionTrees( eval, scenein, h_gradval, groundz, patch_id_hash, aFrame, aTime );
    }
    else AddContributionFarms( eval, scenein, h_gradval, groundz, patch_id_hash, closest_path_middle_point_height );

    if ( BUSH ) AddContributionBushes( eval, scenein, h_gradval, groundz, patch_id_hash, is_forest_patch, aTime, aFrame );
    if ( GRASS ) AddContributionGrass( eval, p, groundz, patch_id_hash );
    if ( PATH ) AddContributionPath( eval, scenein, groundz );

    return eval;
}

struct TraceOutput
{
    float t;
    float dist; // distance to surface (error)
    float shadow; // sun/main light occlusion
};

#define MAX_ITERATIONS_VIEW 120
#define MAX_ITERATIONS_SHADOW 40 // set this as small as you can with your lighting setting, even if shadow ray escape to sky quickly this results in big win
#define TMAX_VIEW 80.0 // was 200 before
#define TMAX_SHADOW 40.0 // reducing this doesn't help much
#define TFRAC 0.8
#define DBREAK 0.0025 // tweak for perfs!!! depends on scene scale etc might make small features thicker than they actually are

TraceOutput traceScene( Ray ray, float shadow_sharpness, int trace_flags
                        , int max_iterations, float tfrac, float tmaxmax, float dbreak, vec2 uv, vec2 fragCoord, int aFrame, float aTime )
{
    TraceOutput to;
    to.t = 0.0;
    to.dist = 0.0;
    to.shadow = 1.0;

    float tmax = tmaxmax; // default to absolute max

    {
        // clamp traced segment
        float hmax = MAX_TERRAIN_HEIGHT + TALLEST_OBJECT_HEIGHT; // there must be nothing above this height
        float thit = plane_trace_z( ray, hmax, 1e-4 );
        if ( thit > 0.0 )
        {
            if ( ray.o.z > hmax ) to.t = thit; // above hmax looking down
            else tmax = min( thit, tmaxmax ); // below hmax looking up, clamp at hmax
        }
        else if ( ray.o.z > hmax )
        {
            to.t = tmaxmax * 1.1; // above hmax looking up, there is only sky
//          return to; // don't branch here, that might be actually slower
        }
    }

    SceneIn scenein;
    SetSceneInDirection( scenein, ray.o, ray.d, trace_flags );
    scenein.t0 = to.t;
    
    for ( int i = 0 FORCE_LOOP; i < max_iterations; ++i )
    {
        scenein.p = ray.o + to.t * ray.d;
        scenein.t = to.t;
        SceneOut eval = evalScene( scenein, aFrame, aTime );
        float d = min( eval.object_di.d, eval.d_ghost );
        // note: ghost points might make us jump over solid tfrac points
        float is_ghost_step = ( d == eval.d_ghost ? 1.0 : 0.0 );
        to.dist = d;

        // important: do not move this block after the to.dist check!
        if ( IsShadowTrace( trace_flags )
            //&& ( is_ghost_step == 0.0 ) // creates ugly discontinuities
            )
        {
            // note: if eval.object_di.d < 0 we set shadow to 0 in effect
            // that catches the case where first point is inside an object (because shadow ray offset issues, ex: bush vs ground normal discontinuity)
            // for regular case if distance is neg it means we hit an object and so shadow = 0 too anyway
            // http://www.iquilezles.org/www/material/nvscene2008/rwwtt.pdf for shadows
            to.shadow = min( to.shadow, shadow_sharpness * max( eval.object_di.d, 0.0 ) / ( to.t + 1e-4f ) );
            
            if ( to.shadow <= 0.01 ) break;
        }

        // warning: never stop on a ghost step!!
        if ( ( ( is_ghost_step == 0.0 ) && ( to.dist <= dbreak * to.t ) )
             || ( to.t > tmax ) ) break;

        // the amount by which we advance t: drop tfrac on ghost steps
        float dt = to.dist * mix( tfrac, 1.0, is_ghost_step );

        to.t += dt;
    }

    if ( to.t > tmax ) to.t = tmaxmax * 1.1;

    return to;
}

#ifdef SHADERTOY_STANDALONE
#define TIME_OF_DAY iSlider0
#else
#define TIME_OF_DAY 0.56
#endif

vec3 get_sun_direction(float aTime)
{
    float sun_elevation = radians( mix( -30.0, 90.0, TIME_OF_DAY ) ); // careful with z, long shadows make the tracing slower
    return zup_spherical_coords_to_vector( unit_vector2( PI * 0.5 - sun_elevation )
                                             , _1 ? V45 : unit_vector2( 2.0 * PI * aTime / 3.0 ) );
}

#define cloud_re 3000.0
#define cloud_r1 ((cloud_re)+8.0)
#define cloud_r2 ((cloud_r1)+1.8)
#define fake_earth_center vec3(0,0,-cloud_re)

float cloudDensity( vec3 p )
{
    float sparsity = 0.07; // 0,1.x
    float freq = 0.2;
    float d = efbm4_13( p * freq, 1.0 + sparsity );
    float r = length( p - fake_earth_center );
    float r2 = length( p.xy );
    float c = smoothstep( 3., 5., r2 ); // cleanup singularity at north pole a little bit
    d *= smoothstep( 0.4, 0.6, d * c ); // multiply d by c give best fade out
    d = max( d, 0. );
    d *= smoothstep( cloud_r1, cloud_r1 + 0.1, r )
        * ( 1.0 - smoothstep( cloud_r2 - 0.1, cloud_r2, r ) ); // altitude band
    return d;
}

// make some ultra basic clouds out of thin air, we will bake them in a spheremap
vec3 traceClouds( vec3 n, float aTime )
{
    vec3 sun_direction = get_sun_direction(aTime);
    Ray ray = mkray( vec3( 0.0 ), n );
    vec2 vt1 = sphere_trace( ray, cloud_r1, fake_earth_center );
    vec2 vt2 = sphere_trace( ray, cloud_r2, fake_earth_center );
    //return vec4( vec3( 0.1*( t2.y - t1.y ) / ( r2 - r1 ) ), 0 );
//  float li = 1.0;
    float vlen = ( vt2.y - vt1.y );
    float vdt = vlen / 100.0;
    float vt = vt1.y;
    float vod = 0.0;
    float vod2 = 0.0;
    float c = 0.8;
    // view ray
    for ( ; vt < vt2.y ; vt += vdt )
    {
        vec3 p = ray.o + ray.d * vt;
        float pd = cloudDensity( p );
        {
            // sun ray
            Ray lray = mkray( p, sun_direction );
            vec2 lt2 = sphere_trace( lray, cloud_r2, fake_earth_center );
            float llen = lt2.y;
            float ldt = llen / 5.0;
            float lt = ldt;
            float lod = 0.0;
            for (; lt < llen; lt += ldt )
            {
                vec3 lp = lray.o + lray.d * lt;
                float lpd = cloudDensity( p );
                lod += vdt * lpd * c * 10.0;
            }
            vod2 += vdt * pd * exp( -lod -vod ); // inscatter
        }
        vod += vdt * pd * c; // absorption
    }

    // for compositing we do something like exp( -vod ) * distant_sky_color + vod2
    return vec3( saturate(vod2), exp( -vod ), 0 );
    
#define cloud_lowest_absorption_remap 0.6
    
}

// map theta to uv .5 r, we only keep the [0,PI/2] theta range
#if 0
float theta2r05( float theta ) { return theta *(1.0/PI) ; }
float r052theta( float r ) { return r * PI; }
#else
// give more resolution to the horizon
float theta2r05( float theta ) { return pow(saturate(theta*(1.0/(PI*0.5))),2.0) *0.5; }
float r052theta( float r ) { return pow((r*2.0),0.5) *PI*0.5; }
#endif

vec3 get_cloud( vec3 v, sampler2D aChannel1, vec3 aResolution, float aTime )
{
    vec3 cloud = vec3(0.0);
#if BUFFER_MODE==2
    if ( CLOUD_MODE>0 )
    {
        vec2 sc = vector_to_zup_spherical_coords( v );
        if ( sc.x > PI * 0.5 ) return BLUE;
        vec2 uv = vec2( 0.5 ) + unit_vector2( sc.y ) * theta2r05( sc.x );
        uv.x *= aResolution.y / aResolution.x;
        cloud = CLOUD_MODE==1
            ? traceClouds( v, aTime )
            : texture( aChannel1, uv ).xyz; // return the highres one here for debug, should match with sampled one
    }
#endif  
    return cloud; // debug
}

float sFlarePeak( vec2 p, float da, float a, float a_offset )
{
    a += a_offset;
    a = floor( ( a / da ) + 0.5 ) * da;
    vec2 vv = unit_vector2( a - a_offset );
    return abs( dot( p, perp( vv ) ) );
}

vec4 sunGlareCoords( mat4 cam, vec3 v, vec3 l )
{
    vec3 sy = normalize( cross( cam[0].xyz, l ) );
    vec3 sx = normalize( cross( l, sy ) );
    return vec4( normalize( vec2( dot( v, sx ), dot( v, sy ) ) ), dot( v, l ), -cam[2].z );
}

vec3 sunGlare( vec4 ppd // xy: angle as unit vector z: dot(v,l)
               , float solid_sun_disk_radius
               , float ray_length // higher value = shorter length
               , float ray_thickness
               , float ray_thickness_disparity
               , float n1 // can be arbitrarily large
               , float n2 // there is a loop on n2 - the number of rays is n1*n2 so we have a trade off between the repeat in sFlarePeak (which can't do proper additive on rays) and the for loop (which can)
               , float falloff_glare_attn
               , float falloff_glare_p1
               , float falloff_glare_p2
               , bool falloff_glare_multiplicative )
{
    vec2 anglev = ppd.xy; // angle as unit vector
    float v_dot_l = ppd.z;
    float r = safe_acos( v_dot_l ) / PI; // could do without acos but it is a lot more easier to work with regular spacing
    vec2 p = anglev * r;
    float a = calc_angle( anglev );
//  return vec3( stripes(r,0.02,0.001,0.001/2.0), stripes(degrees(a),10.,0.1,0.1/2.0),0.); // visualize polarcoords
    float v = 0.0;
    float d2 = max( r - solid_sun_disk_radius, 0. ); // falloff glare
    float da = 2.0 * PI / n1;
    float da2 = 2.0 * PI / ( n1 + n2 );
    for ( float i = 0.; i < n2; ++i )
    {
        vec3 rr = hash31( i + 1. );
        float rda = rr.x * da2 * 0.1;
        float d1 = sFlarePeak( p, da, a, -i * da / n2 + rda );
        if ( r < solid_sun_disk_radius ) d1 = 0.;
        v += exp2( -d1 * d1 * ray_thickness * mix( 1.0-ray_thickness_disparity, 1.+ray_thickness_disparity, rr.y )
                   -d2 * ray_length * mix( 0.5, 1.5, rr.z ) );
    }
    float falloff_glare = falloff_glare_attn * ( 1.0 - powerful_scurve( d2, falloff_glare_p1, falloff_glare_p2 ) );
    v = falloff_glare_multiplicative ? v * falloff_glare : v + falloff_glare;
    v = min( v, 1. );
    return vec3( v );
}

#define FOG_GROUND _1

 // set render to false for getting a value for lighting calculation as opposed to display
vec3 get_top_sky_color( bool render ) {  return AZURE * ( render ? 0.6 : 1.0 ); }
vec3 get_sky_horizon_color() { return mix( WHITE, AZURE, 0.4 ); }
vec3 get_fog_color() { return mix( AZURE, WHITE, 0.15 ); }

// return a v.z normalized so that horizon view direction is remapped to 0 and zenith is still 1
float get_hacked_vz( vec3 v, float ez )
{
    float zmin = -max(ez,0.)/(TMAX_VIEW*1.1); // z/f=zmin/1. where f=TMAX_VIEW*1.1
    return max(0.,(v.z-zmin)/(1.-zmin));
}

vec3 get_sky( vec3 v, vec3 l, float ez, bool render )
{   
    float hz = get_hacked_vz( v, ez );
    return mix( get_sky_horizon_color(), get_top_sky_color(render), pow( hz, 0.2 ));
}

vec3 get_sky_plus_clouds( vec3 v, vec3 l, float ez, sampler2D aChannel1, vec3 aResolution, float aTime )
{
    vec3 col = get_sky( v, l, ez, true );
    vec3 cloud = get_cloud( v, aChannel1, aResolution, aTime  );
    vec3 sky_with_clouds =  col * mix( cloud_lowest_absorption_remap, 1., cloud.y ) + vec3( cloud.x ); // composite clouds, col * absorption + scattering
    return mix( col, sky_with_clouds, smoothstep( 0., 0.025, v.z ) ); // fade to sky on thin horizon band
}

// e = eye pos, v = view vector, p = lit point, n = normal, l = sun direction
vec3 shadeSample( in vec3 e, in vec3 v, mat4 cam, in vec3 p, in vec3 n, vec3 l
                  , float shadow, TraceOutput to, float ao, vec2 uv
                  , SceneOut eval, bool sky, sampler2D aChannel1, vec3 aResolution, float aTime )
{
    vec3 col = vec3( 0. );
    
    vec3 sunI = vec3( 2.0 );

    vec3 top_sky_color = get_sky(vec3(0,0,1),l,e.z,false);

    SG sun_lobe = CosineLobeSG( l );
    sun_lobe.Amplitude *= sunI;

    SG sky_lobe = CosineLobeSG( vec3( 0, 0, 1 ) );
    sky_lobe.Amplitude *= top_sky_color;

    if ( !sky )
    {
        vec3 scene_color = vec3( 0 );
    
        float bush_factor = 0.0;
        float tree_factor = 0.0;
        float pine_factor = 0.0;
        vec3 sky_reflection = vec3(0,0,0);
        
        vec2 mm = DecodeId_16( eval.object_di );
        float matid = mm.x;
        float matid_f = mm.y; // we can have color variations within an id

        bush_factor = smoothbump( MATID_BUSH, 0.5, matid );
        tree_factor = smoothbump( MATID_TREE, 0.5, matid );
        pine_factor = smoothbump( MATID_PINE, 0.5, matid );
        vec4 mm4 = DecodeId_5_5_5( eval.object_di ); // yzw = depth uheight color_rnd
        float tree_ao = mix( 0.6, 1., smoothstep( 0., 0.5, mm4.z ) ); // make it very faint, doesn't fit the style

        scene_color = vec3( 0 )
            + bush_factor * eval.color
            + tree_factor * mix( mix( COLOR_TREE1, COLOR_TREE2, mm4.w ), COLOR_TREE_SURF, 0.0*mm4.y*mm4.y*mm4.y ) * tree_ao
            + smoothbump( MATID_TRUNK, 0.5, matid ) * COLOR_TRUNK
            + smoothbump( MATID_HOUSE, 0.5, matid ) * COLOR_HOUSE
            + smoothbump( MATID_HOUSE_BOTTOM, 0.5, matid ) * COLOR_HOUSE_BOTTOM
            + smoothbump( MATID_ROOF, 0.5, matid ) * mix( COLOR_ROOF1, COLOR_ROOF2, matid_f )
            + smoothbump( MATID_BRICKWALL, 0.5, matid ) * COLOR_BRICKWALL
            + pine_factor * mix( COLOR_PINE, COLOR_PINE2, mm4.w ) * tree_ao
            + smoothbump( MATID_GRASS, 0.5, matid ) * mix( COLOR_GRASS, COLOR_GRASS2, matid_f )
            + smoothbump( MATID_GROUND, 0.5, matid ) * mix( PATH_COLOR, PATH_COLOR2, matid_f )
            + smoothbump( MATID_NONE, 0.5, matid ) * MAGENTA;

        if ( MATID_WINDOW == matid ) 
        {
            scene_color = BLACK;
            vec3 vr = reflect( v, n );
            vec3 refl_color = get_sky_plus_clouds( vr, l, p.z, aChannel1, aResolution, aTime );
            refl_color = mix( COLOR_GRASS * mix( 0.2, 1.0, shadow ), refl_color, smoothstep( -0.03, -0.0, vr.z ) );
            sky_reflection = refl_color * mix( 0.2, 1.0, pow( 1.0 - saturate( dot( -v, n ) ), 2.0 ) );
        }

        if ( MATID_GRASS == matid )
        {
            vec3 lf = sfbm4_33( p * 40.0 );
            float hf = sfbm1_12( p.xy * 2800.0 );

            if ( matid_f > ( 1. - 0.04 ) ) // mowed lawn, should be rare
            {
                float period = 0.07;
                float s = stripes( rotate_with_angle( p.xy, matid_f * 1000.0 ).x, period, period * 0.25, 0.004 );
                scene_color = mix( COLOR_MOWED_GRASS, COLOR_MOWED_GRASS2, saturate( s + ( lf.x - 0.38 ) * 1.5 ) );
            }

            // just apply some noise
            scene_color *= 1.0 - 0.2 * ( saturate( 1.0 - lf.y ) );
            scene_color *= saturate( 1.0 - 0.4 * hf * saturate( 1.0 - lf.y ) );
        }
    
        vec3 albedo = scene_color; // return albedo;

        col += shadow * SGDiffuseFitted( sun_lobe, n, albedo );

        // this way of doing diffuse makes shadow color be different on ground than back of object resting on it which sucks
        if ( _1 ) col += mix( SGDiffuseFitted( sky_lobe, n, albedo ), top_sky_color, SHADOW_TINT_SATURATION )
                // add more sky ambient to tint shadow in blue? in a way that is not too hacky?
                * mix( 0.15, 0.02, shadow )
                * ao;

        col += sky_reflection;

        float d = length( p - e );
        if ( FOG_GROUND ) col = mix( col, get_fog_color() * 0.8, ( 1. - exp( -0.1 * max( d - 6.1, 0. ) ) ) * 0.378 ); // fog
    }
    else
    {
        col = get_sky_plus_clouds( v, l, e.z, aChannel1, aResolution, aTime );

        if ( SUN ) // this sun is the one in the sky
        {
            vec4 ppd = sunGlareCoords( cam, v, l );
            col += sunGlare( ppd, 0.007, 280., 100000.0, 0.2, 20.0, 4.0, 0.3, 0.4, 8., false )
                * vec3( 0.8, 0.8, 0.5 ) * sunI;
        }
    }

    // note: we would like to do SUN_GLARE here, but reprojection only works on solid world pos or distance sky
    return col;
}

vec3 postProcess( vec3 col, vec2 uv, vec4 ppd )
{
    if ( SUN_GLARE )
    {
        col.xyz += 0.116 * sunGlare( ppd, 0., 7., 8000.0, 0.2, 8.0, 8.0, 0.3, 2., 2., true )
            * vec3( 1., 0.7, 0.2 ) * 2. // use warmer color for this glare
            * smoothstep( -0.15, 0.3, ppd.w )
            * (1.0-smoothstep( 0.78, 0.9, ppd.z ));
    }

    float exposure = 3.0; 
    // maybe auto expose when staring straight at the sun? the sun is a bit saturated on the clouds
    //exposure = mix(exposure,1.,smoothstep(0.985,1.0,1.-ppd.z));
    col = exposure * tonemap_reinhard( col );
//  col = max( vec3( 0. ), contrast( col, vec3( 1.02 ) ) );
    float vignette = .0 + 1. * pow( 20. * uv.x * uv.y * ( 1. - uv.x ) * ( 1. - uv.y ), 0.15 );
    col *= vignette;
    col = contrast( col, vec3( 1.06 ) );
    col = gamma_correction_itu( col );
    return col;
}

struct CameraRet { vec3 eye; vec3 target; float roll; float pitch; };

CameraRet init_cam() { CameraRet cam; cam.roll = 0.; cam.pitch = 0.; return cam; }

#define STICKY_MOUSE false

mat4 look_around_mouse_control( mat4 camera, float pitch, float tan_half_fovy, vec3 aResolution, vec4 aMouse, float dmmx )
{
    float mouse_ctrl = 1.0;
    vec2 mm_offset = vec2( dmmx, pitch );
    vec2 mm = vec2( 0.0, 0.0 );

#ifndef EXTRA_3D_CAMERA
    if ( aMouse.z > 0.0 || STICKY_MOUSE ) mm = ( aMouse.xy - aResolution.xy * 0.5 ) / ( min( aResolution.x, aResolution.y ) * 0.5 );
#endif

    mm.x = -mm.x;
    mm = sign( mm ) * pow( abs( mm ), vec2( 0.9 ) );
    mm *= PI * tan_half_fovy * mouse_ctrl;
    mm += mm_offset;

    return camera * yup_spherical_coords_to_matrix( mm.y, mm.x );
}

struct LanePoint { vec3 p; vec3 tangent2d; float lane_index; };

LanePoint getPathPointAndDir( float x, float lane_index )
{
    LanePoint ret;
    vec2 y = wavyLaneFuncAndDerivative_WORLD( x, lane_index );
    ret.p.xy = vec2( x, y.x );
    ret.p.z = BaseGroundHeight( ret.p.xy ) + 0.073;
    ret.tangent2d = normalize( vec3( 1.0, y.y, 0 ) );
    ret.lane_index = lane_index;
    return ret;
}

// a cubicstep function that takes 2 arbitrary end points and 2 (begin end) slopes
float cubicstep2( float x, vec2 p0, vec2 p1, float s0, float s1 )
{
    x -= p0.x;
    p1 -= p0;
    x = clamp( x, 0., p1.x );
    float x1_sqr = p1.x * p1.x;
    vec3 eq1 = vec3( 3.0 * x1_sqr, 2.0 * p1.x, s1 - s0 );
    vec3 eq2 = vec3( x1_sqr * p1.x, x1_sqr, p1.y - s0 * p1.x );
    float a = ( eq1.y * eq2.z - eq1.z * eq2.y ) / ( eq1.y * eq2.x - eq1.x * eq2.y );
    float b = ( eq1.z - eq1.x * a ) / eq1.y;
    return p0.y + ( ( a * x + b ) * x + s0 ) * x;
}

float linearstep2( float x, vec2 p0, float s0 ) { return p0.y + s0 * ( x - p0.x ); }

// getCameraPathx is a line segment of length 'a', slope 'sa' followed by another line segment of length 'b', slope 'sb',
// made periodic and smooth. 
// the 'a' segment is where we walk, the 'b' segment is where we fly 
// this lets us tweak length and speed of walk and flight sequences separately
// the cubic transitions are a bit iffy and tedious looking but the curve must be really clean to make landing and take off transitions natural and gentle
// return values:
//  x: were we are on the lane
//  y: flight smooth blend factor (0 means we walk >0 means we fly, function goes up and down smootly)
//  z: the periodicity index, so each period can use a different lane
//  w: the flight begin to end smoothstep, we use it so we can blend between lanes whilst we are in the air
vec4 getCameraPathx( float x, float a, float b, float sa, float sb, float e )
{
    e = min( e, min( a, b ) * 0.5 ); // e is the half lenght of smooth transitions
    float period = a + b;
    float h = sa * a + sb * b;
    float n = floor( x / period );
    x -= n * period;
//#if 1
    vec2 M = vec2( a, sa * a );
    vec2 F = vec2( a + b, h );
    vec2 A = vec2( 0, 0 );
    vec2 B = A + vec2( e, sa * e );
    vec2 C = M - vec2( e, sa * e );
    vec2 D = M + vec2( e, sb * e );
    vec2 E = F - vec2( e, sb * e );
    float y = n * h;
    if ( x < B.x ) y += cubicstep2( x, A + ( E - F ), B, sb, sa );
    else if ( x < C.x ) y += linearstep2( x, B, sa );
    else if ( x < D.x ) y += cubicstep2( x, C, D, sa, sb );
    else if ( x < E.x ) y += linearstep2( x, D, sb );
    else y += cubicstep2( x, E, F + ( B - A ), sb, sa );
    float section_b = min( smoothstep( C.x, D.x, x ), 1. - smoothstep( E.x, F.x + B.x - A.x, x ) )
    + 1.0 - smoothstep( A.x + E.x - F.x, B.x, x );
    return vec4( y, section_b, n, smoothstep( D.x, E.x, x ) ); // w: flight begin and end for lane transitions
//#endif
//  // the function above if it was just piecewise linear
//  if ( x < a ) x *= sa; else x = sa * a + ( x - a ) * sb;
//  return vec4( x + n * h, 0, n, 0 );
}

// get the args for getCameraPathx, with a travel lengh l and speed in input
vec2 get_duration_and_slope( float l, float speed ) { float t = l / speed; return vec2( t, sqrt( max( l * l - t * t, 0. ) ) / t ); }

mat4 walkAndFlyCamera( float tan_half_fovy, float aTime, vec3 aResolution, vec4 aMouse )
{
#if 1
    vec2 aa = get_duration_and_slope( 2.82, 1.41 );
    vec2 bb = get_duration_and_slope( 11.34, 7.56 );
    float te = 0.4;
    vec4 vv = getCameraPathx( te + aTime * 0.07, aa.x, bb.x, aa.y, bb.y, te ); // vv.y = is fly mode amount
    float lane_index = FIRST_LANE_INDEX;
    lane_index += vv.z;
    LanePoint pp_next = getPathPointAndDir( vv.x, lane_index+1.0 );
    LanePoint pp = getPathPointAndDir( vv.x, lane_index );
    // lerp with the next lane
    pp.p = mix(pp.p,pp_next.p,vv.w);
    pp.lane_index = mix(pp.lane_index,pp_next.lane_index,vv.w);
    pp.tangent2d = normalize(mix(pp.tangent2d,pp_next.tangent2d,vv.w));
    // lerp with straight curve corresponding to that (now blending) lane
    float alt = 1.;
    pp.p.z = mix( pp.p.z, alt, vv.y * vv.y );
    pp.p.y = mix( pp.p.y, ( pp.lane_index + 0.5 ) * PATH_SPACING_Y, vv.y * vv.y );
    pp.tangent2d = normalize( mix( pp.tangent2d, vec3( 1., 0., 0. ), vv.y ) );
#else
    LanePoint pp = getPathPointAndDir( 9.8+aTime * 0.08, FIRST_LANE_INDEX ); // simple walk camera
#endif
    CameraRet cam = init_cam();
    cam.eye = pp.p;
    cam.target = cam.eye + pp.tangent2d;
    cam.target.z -= mix(0.15,0.53, vv.y * vv.y); // this causes some roll on mouse movements as a happy side effect
//  cam.target.z -= 0.59 * vv.y * vv.y; // this causes some roll on mouse movements as a happy side effect
    mat4 camera = lookat( cam.eye, cam.target, vec3( 0., 0., 1. ) ) * z_rotation( cam.roll );
    float tilt = 0.2 * sin( aTime * 0.02 ) * vv.y * vv.y; // in flight mode add some small left-right to get some tilt when no one touches the mouse
    vec2 dmm = vec2( tilt, -0.35 * pow( abs( tilt ), 0.6 ) );
    return look_around_mouse_control( camera, cam.pitch + dmm.y, tan_half_fovy, aResolution, aMouse, dmm.x );
}

vec2 pixelIndexToFragcoord( vec2 pixel_indexf, vec3 aResolution )
{
    // note that pixelIndexToFragcoord(floor(fragCoord))==fragCoord
    return aResolution.xy * ( ( vec2( 0.5 ) + pixel_indexf ) / aResolution.xy );
}

struct CameraData { mat4 camera; float tan_half_fovy; }; // what we need to write for the reprojection

CameraData GetCameraTransform(float aTime, vec3 aResolution, vec4 aMouse)
{
    CameraData data;

    data.tan_half_fovy = 0.6;

#ifdef EXTRA_3D_CAMERA
    data.camera = mat4( iCamera[0], iCamera[1], iCamera[2], iCamera[3] );
    data.tan_half_fovy = iTanHalfFovy;
#else
    data.camera = walkAndFlyCamera( data.tan_half_fovy, aTime,aResolution, aMouse );
#endif
    return data;
}

vec3 mat_project_vector( vec3 v, mat4 camera ) { return vec3( dot( v, camera[0].xyz ), dot( v, camera[1].xyz ), dot( v, camera[2].xyz ) ); }
vec3 mat_project_point_dir( vec3 p, mat4 camera ) { return normalize( mat_project_vector( p - camera[3].xyz, camera ) ); }

// get subpixel index, 2x2 for now
ivec2 SubpixelIndex( int aFrame ) { aFrame &= 3; return ivec2( aFrame & 1, aFrame >> 1 ); }

vec2 NumSubpixels( vec3 aResolution ) { return aResolution.xy / 2.0; }

CameraData readCameraData( ivec2 offset, vec3 aResolution, sampler2D aChannel1 )
{
    CameraData data;
#if 0
    float y = offset.x + 2.0 * offset.y;
    data.camera[0] = vec4( texture( aChannel1, ( vec2( 0.5 ) + vec2( 0, y ) ) / aResolution.xy, 0 ).xyz, 0 );
    data.camera[1] = vec4( texture( aChannel1, ( vec2( 0.5 ) + vec2( 1, y ) ) / aResolution.xy, 0 ).xyz, 0 );
    data.camera[2] = vec4( texture( aChannel1, ( vec2( 0.5 ) + vec2( 2, y ) ) / aResolution.xy, 0 ).xyz, 0 );
    data.camera[3] = vec4( texture( aChannel1, ( vec2( 0.5 ) + vec2( 3, y ) ) / aResolution.xy, 0 ).xyz, 1 );
    data.tan_half_fovy = texture( aChannel1, ( vec2( 0.5 ) + vec2( 4, y ) ) / aResolution.xy, 0 ).x;
#else
    int y = offset.x + 2 * offset.y;
    data.camera[0] = texelFetch( aChannel1, ivec2( 0, y ), 0 );
    data.camera[1] = texelFetch( aChannel1, ivec2( 1, y ), 0 );
    data.camera[2] = texelFetch( aChannel1, ivec2( 2, y ), 0 );
    data.camera[3] = texelFetch( aChannel1, ivec2( 3, y ), 0 );
    data.tan_half_fovy = texelFetch( aChannel1, ivec2( 4, y ), 0 ).x;
#endif
    return data;
}

vec4 mainScene( vec2 fragCoord, float aTime, vec3 aResolution, int aFrame, vec4 aMouse, sampler2D aChannel1
                , out vec4 ppd )
{
    float aspect = aResolution.x / aResolution.y;
    vec2 uv = fragCoord.xy / aResolution.xy;

    vec2 pixel = fragCoord.xy;
    vec4 fragColor = vec4( 0.0 );

#if 1
    CameraData camera_data = GetCameraTransform( aTime, aResolution, aMouse );
#else
    ivec2 offset0 = SubpixelIndex( aFrame - 0 ); // I think we get previous frame because double buffer so doesn't work as is
    CameraData camera_data = readCameraData( offset0, aResolution, aChannel1 );
#endif

    Ray view_ray = get_view_ray2( ( uv - vec2( 0.5 ) ) * 2.0, aspect, 1.0 / camera_data.tan_half_fovy, camera_data.camera );
    float ao = 1.0;

    vec3 sun_direction = get_sun_direction( aTime );
    bool sky = false;
    float shadow = 1.0;
    TraceOutput to;
    vec3 p2;
    vec3 n;
    SceneOut eval_for_shade;
    SceneOutInit( eval_for_shade );
    Ray trace_ray = view_ray;
    int trace_flags = TRACE_VIEW;
    int max_iterations = MAX_ITERATIONS_VIEW; 
    float tmaxmax = TMAX_VIEW;

    // force a loop on view ray, shadow ray to prevent shadertoy compilation abject unrolling horror
    for ( int pass = 0 FORCE_LOOP; pass < (SHADOWS?2:1); ++pass )
    {
        TraceOutput tmp_to = traceScene( trace_ray, 5.0, trace_flags, max_iterations, TFRAC, tmaxmax, DBREAK
                                         , uv, fragCoord, aFrame, aTime );
        if ( pass == 1 )
        {
            //shadow = ( tmp_to.t > TMAX_SHADOW ? 1 : 0 ); // check hard shadows
            //shadow = tmp_to.shadow; // default soft shadows, don't work well with this scene
            shadow = mix( 0.05, 1.0, smoothstep( 0.4, 0.6, tmp_to.shadow ) ); // take a threshold on default soft shadows, good for sunny setting
            break;
        }

        // only view rays make it here... 

        to = tmp_to;
        p2 = view_ray.o + to.t * view_ray.d;
        sky = to.t > TMAX_VIEW;

        if ( sky ) break;
        
        // only view rays that hit solid surfaces make it here...

        // we need to evaluate normal vector at hit point, we will also retrieve extra material calculations
        SceneIn scenein;
        scenein.p = p2;
        SetSceneInDirectionless( scenein, TRACE_SHADE );
        scenein.t = 0.0;

        {
            // to set this epsilon, set the camera at 1000 and check fence and terrain normal...
            // it should look the same as 0,0,0...
            float e = 1e-3 * 2.;
            vec4 v; // center in .w, deltas in .xyz
#if 0
            SceneIn scenein2 = scenein;
            scenein2.p = scenein.p + vec3( e, 0.0, 0.0 ); v.x = evalScene( scenein2, aFrame, aTime ).object_di.d;
            scenein2.p = scenein.p + vec3( 0.0, e, 0.0 ); v.y = evalScene( scenein2, aFrame, aTime ).object_di.d;
            scenein2.p = scenein.p + vec3( 0.0, 0.0, e ); v.z = evalScene( scenein2, aFrame, aTime ).object_di.d;
            eval_for_shade = evalScene( scenein, aFrame, aTime );
            v.w = eval_for_shade.object_di.d;
#else
            // force a loop on gradient eval to prevent shadertoy compilation abject unrolling horror
            for ( int i = 0 FORCE_LOOP; i < 4; ++i )
            {
                SceneIn scenein2 = scenein;
                if ( i != 3 ) scenein2.p[i] += e; // let's live dangerously and use vector component random access
                eval_for_shade = evalScene( scenein2, aFrame, aTime ); // note: eval_for_shade contains material data at center, at the end of the loop
                v[i] = eval_for_shade.object_di.d;
            }
#endif
            n = normalize( v.xyz - vec3( v.w ) );
        }

        if ( AO )
        {
            SceneIn scenein2 = scenein;
            SetSceneInDirection( scenein2, p2, n, TRACE_AO );  // shouldn't change anything

            // http://www.iquilezles.org/www/material/nvscene2008/rwwtt.pdf
            float delta = 0.1;
            float a = 0.0;
            float b = 1.0;
            for ( int i = 0 FORCE_LOOP; i < 4; i++)
            {
                float fi = float( i );
                scenein2.p = p2 + n * delta * fi;
                float d = evalScene( scenein2, aFrame, aTime ).object_di.d;
                a += ( delta * fi - d ) * b;
                b *= 0.5;
            }
            ao = max( 1.0 - 1.8 * a, 0.0 );
        }

        if ( SHADOWS )
        {
            // hack: use a different normal offset for trees as noise shadows are very sensitive to that
            vec4 mm4 = DecodeId_5_5_5( eval_for_shade.object_di );
            float tree_ao = 1.0;
            if ( (mm4.x == MATID_TREE) || (mm4.x == MATID_PINE) )
            {
                tree_ao += 1.0 - smoothstep( 0.2, 0.5, mm4.z );
                tree_ao -= smoothstep( -0.2, 0.0, dot( sun_direction, n ) );
                tree_ao = saturate( tree_ao );
            }

            // note: because of surface noise, tweaking the shadow ray normal bias has a lot of impact on vegetation
            // might want to tweak it per surface too
            trace_ray = mkray( p2 + n * mix( 0.004, 0.0005, tree_ao ), sun_direction ); // if bias is too small here tree shadows become shitty
            trace_flags = TRACE_SHADOW;
            max_iterations = MAX_ITERATIONS_SHADOW;
            tmaxmax = TMAX_SHADOW;
        }
    }

    fragColor.rgb = shadeSample( view_ray.o, view_ray.d, camera_data.camera, p2
                                 , n, sun_direction, shadow, to, ao, uv, eval_for_shade, sky, aChannel1, aResolution, aTime );

    ppd = sunGlareCoords( camera_data.camera, view_ray.d, sun_direction );

    fragColor.a = to.t; // write depth in .w to recover the world position in reprojection

    return fragColor;
}

#if BUFFER_MODE==0
vec4 Mode0_NoBuffer_mainImage( vec2 fragCoord, float aTime, vec3 aResolution, int aFrame, vec4 aMouse, sampler2D aChannel1 )
{
    vec4 ppd;
    vec4 col = mainScene( fragCoord, aTime, aResolution, aFrame, aMouse, aChannel1, ppd );
    col.rgb = postProcess( col.rgb, fragCoord / aResolution.xy, ppd );
    return col;
}
#endif

#if BUFFER_MODE==2

// write camera
vec4 Mode2_Reproject_mainBufferB( vec2 fragCoord, float aTime, vec3 aResolution, int aFrame, vec4 aMouse, sampler2D aChannel0, sampler2D aChannel1 )
{
    vec2 pi = floor( fragCoord );
    vec4 background = texelFetch( aChannel1, ivec2( pi ), 0 );

    if ( ( pi.y > 3.0 ) || 
         ( pi.x > 4.0 ) ) 
    {
        vec4 skybox = background;

        ivec2 res_addr = ivec2( 5, 0 ); // store position for resolution

        if ( ( CLOUD_MODE > 1 ) &&
             ( ( CLOUD_MODE == 2 ) // always
               || ( aFrame <= 1 ) // on first frame
               || ( texelFetch( aChannel1, res_addr, 0 ).xy != vec2( aResolution.xy ) ) ) ) // on resolution change
        {
            if ( pi == vec2(res_addr) ) return vec4( aResolution.xy, 0, 0 );
            
            vec2 p = fragCoord / aResolution.y;
            if ( p.x > 1. || p.y > 1. ) skybox.xyz = MAGENTA;
            else
            {
                skybox.xyz = vec3( p,0 ); //vec3( traceClouds( n ) );
                p -= vec2( 0.5 );
                float r = length( p );
                vec3 n = zup_spherical_coords_to_vector( r052theta( r ), calc_angle( p ) ); // only care about theta in [0,PI/2]
                skybox.xyz = traceClouds( n, aTime );
                if ( r > 0.5 ) skybox.xyz = RED;
                //skybox = vec4( get_sky( n, vec3( 0 ) ).w, 0 );
            }
        }

        if ( CLOUD_MODE <= 1 ) skybox.xyz = MAGENTA;

        return skybox; // unused area, we use the bottom (3*4,4) pixels
    }

    vec2 offset = vec2( SubpixelIndex( aFrame ) );
    float y = offset.x + 2.0 * offset.y; // subpixel flat index
    
    if ( pi.y == y )
    {
        CameraData data = GetCameraTransform(aTime,  aResolution, aMouse);
        if ( pi.x == 0.0 ) return vec4( data.camera[0].xyz, 1 );       // return vec4( RED, 1 );    
        if ( pi.x == 1.0 ) return vec4( data.camera[1].xyz, 1 );       // return vec4( GREEN, 1 );  
        if ( pi.x == 2.0 ) return vec4( data.camera[2].xyz, 1 );       // return vec4( BLUE, 1 );   
        if ( pi.x == 3.0 ) return vec4( data.camera[3].xyz, 1 );       // return vec4( YELLOW, 1 ); 
        if ( pi.x == 4.0 ) return vec4( data.tan_half_fovy, 0, 0, 1 ); // return vec4( MAGENTA, 1 );
    }
    return background; // no touch = copy old data - we need to do that because shadertoy double buffers
}

// write each subpixel in its respective quadrant
vec4 Mode2_Reproject_mainBufferA( vec2 fragCoord, float aTime, vec3 aResolution, int aFrame, vec4 aMouse, sampler2D aChannel0, sampler2D aChannel1 )
{
    ivec2 hres = ivec2( aResolution.xy ) / 2;
    ivec2 offset = SubpixelIndex( aFrame );
    ivec2 pi = ivec2( floor( fragCoord ) );
    ivec2 pi0 = pi;
    // gather for coherency (make 4 quadrant, each having a full scene, instead of pixel all top pixels in 4)
    if ( offset != ( pi / hres ) ) // quadrant this pixel is in
    {
        return texelFetch( aChannel0, pi, 0 ); // no touch = copy old data - we need to do that because shadertoy double buffers
    }
    pi = ( pi - offset * hres ) * 2; // also clears offset lower bits
    fragCoord = pixelIndexToFragcoord( vec2( pi + offset ), aResolution );
//  return vec4( offset, 0., 1. ); // quadrant debug color
    vec4 ppd;
    return mainScene( fragCoord, aTime, aResolution, aFrame, aMouse, aChannel1, ppd );
}

// input uv maps 0,1 to screen
vec2 applySubPixelOffsetTo01UV( vec2 uv, vec2 offseti, vec3 aResolution )
{
    // important: since we sample the *center* of the top level pixel,
    // we want the *corner* of the corresponding subpixel in the quadrant image
    // (so offset by half the quadrant offset)
    // without that the image is much blurrier than it should
    // besides, the chance I got this wroing somewhere is high :-D
    uv -= ( offseti - vec2( 0.5 ) ) / aResolution.xy;
    return ( uv + vec2( offseti ) ) * 0.5;
}

float reprojected( vec3 p, ivec2 offseti, CameraData cdati, vec3 aResolution, out vec4 ret, out Ray view_ray, sampler2D aChannel0 )
{
    float aspect = aResolution.x / aResolution.y;
//  CameraData cdati = readCameraData( offseti, aResolution ); // passed from caller
    vec3 v = mat_project_point_dir( p, cdati.camera ); // ray direction in the older camera
    view_ray.d = v;
    view_ray.o = cdati.camera[3].xyz;
    float t = plane_trace_z( v, -1.0 / cdati.tan_half_fovy, 1e-3 );
    vec2 uv = ( v * t ).xy;
    uv.x /= aspect;
    uv = ( uv + vec2( 1.0 ) ) * 0.5;
    vec2 uvcheck = uv; // before going to quadrant uv, store the 01 range uv to do a uv range check see if our sampled thing is valid...
    uv = applySubPixelOffsetTo01UV( uv, vec2(offseti), aResolution );
    if ( ( t < 0.0 ) || ( t == FLT_MAX ) || ( saturate( uvcheck ) != uvcheck ) ) return 0.0;
#if 0
    ret = texture( aChannel0, uv );
#else
#if 0
    ivec2 pmin = ivec2( 0, 0 );
    ivec2 pmax = ivec2( aResolution.xy ) - ivec2( 1, 1 );
#else
    // deal with borders
    ivec2 pmin = offseti * ivec2( aResolution.xy ) / 2;
    ivec2 pmax = pmin + ivec2( aResolution.xy ) / 2 - ivec2( 1, 1 );
#endif
    // bilinear by hand so we can exclude samples and/or clamp borders properly
    vec2 pi = uv * aResolution.xy - vec2( 0.5 );
    ivec2 i = ivec2( floor( pi ) );
    vec2 f = fract( pi );
    vec4 v00 = texelFetch( aChannel0, clamp( i + ivec2( 0, 0 ), pmin, pmax ), 0 ); // return im0;
    vec4 v10 = texelFetch( aChannel0, clamp( i + ivec2( 1, 0 ), pmin, pmax ), 0 );
    vec4 v01 = texelFetch( aChannel0, clamp( i + ivec2( 0, 1 ), pmin, pmax ), 0 );
    vec4 v11 = texelFetch( aChannel0, clamp( i + ivec2( 1, 1 ), pmin, pmax ), 0 );
#if 1
    vec4 A = v10 - v00;
    vec4 B = v01 - v00;
    vec4 C = ( v11 - v01 ) - A;
    vec4 D = v00;
    ret = A * f.x + B * f.y + C * f.x * f.y + D;
#else
    ret = mix( mix( v00, v10, f.x ), mix( v01, v11, f.x ), f.y );
#endif
#endif
//  ret = vec4(uvcheck,0,1);
    return 1.0;
}

// combine images history
vec4 Mode2_Reproject_mainImage( vec2 fragCoord, float aTime, vec3 aResolution, int aFrame, vec4 aMouse, sampler2D aChannel0, sampler2D aChannel1, out vec4 ppd )
{
    ivec2 hres = ivec2( aResolution.xy ) / 2;
    ivec2 pi = ivec2( floor( fragCoord ) );
    ivec2 offset0 = SubpixelIndex( aFrame - 0 ); // this frame
    ivec2 offset1 = SubpixelIndex( aFrame - 1 ); // previous frame
    ivec2 offset2 = SubpixelIndex( aFrame - 2 ); // previous previous frame
    ivec2 offset3 = SubpixelIndex( aFrame - 3 );
    CameraData cdat0 = readCameraData( offset0, aResolution, aChannel1 );
    CameraData cdat1 = readCameraData( offset1, aResolution, aChannel1 );
    CameraData cdat2 = readCameraData( offset2, aResolution, aChannel1 );
    CameraData cdat3 = readCameraData( offset3, aResolution, aChannel1 );
    // reprojection code path
    // we need to be careful here... the point we reproject is the center of the quadrant,
    // the regular raytraced position if we were fullscreen, because it is the only point
    // that is common to all quadrants and the only point that will give us a stable image
    // with 4 last frames (without accumulating)
    vec2 uv = fragCoord / aResolution.xy;
    float aspect = aResolution.x / aResolution.y;
    Ray view_ray0 = get_view_ray2( ( uv - vec2( 0.5 ) ) * 2.0, aspect, 1.0 / cdat0.tan_half_fovy, cdat0.camera );
    vec2 uv0 = applySubPixelOffsetTo01UV( uv, vec2(offset0), aResolution );
    vec4 im0 = texture( aChannel0, uv0 );
//  return im0; // this one is always good, we have just calculated it
    vec3 p0 = view_ray0.o + view_ray0.d * im0.w; // the most recent world point we have
    vec4 im1, im2, im3;
    vec3 valid;
    Ray view_ray1, view_ray2, view_ray3;
    valid.x = reprojected( p0, offset1, cdat1, aResolution, im1, view_ray1, aChannel0 ); vec3 p1 = view_ray1.o + view_ray1.d * im1.w;
    valid.y = reprojected( p0, offset2, cdat2, aResolution, im2, view_ray2, aChannel0 ); vec3 p2 = view_ray2.o + view_ray2.d * im2.w;
    valid.z = reprojected( p0, offset3, cdat3, aResolution, im3, view_ray3, aChannel0 ); vec3 p3 = view_ray3.o + view_ray3.d * im3.w;
    vec4 w = vec4( 1.0 );
    w.yzw = valid.xyz;
    vec4 col = ( im0 * w.x + im1 * w.y + im2 * w.z + im3 * w.w ) / sum( w ); // note: don't preprocess here
    if ( SUN_GLARE ) ppd = sunGlareCoords( cdat0.camera, view_ray0.d, get_sun_direction( aTime ) ); // pass this to post processing
    return col;
}

#endif

#if BUFFER_MODE==0
 #define BUFFERB void mainImage( out vec4 fragColor, in vec2 fragCoord ) { }
 #define BUFFERA void mainImage( out vec4 fragColor, in vec2 fragCoord ) { }
 #define IMAGE void mainImage( out vec4 fragColor, in vec2 fragCoord ) { fragColor = Mode0_NoBuffer_mainImage( fragCoord, iTime, iResolution, iFrame, iMouse , iChannel1); }
#elif BUFFER_MODE==2
 #define BUFFERB void mainImage( out vec4 fragColor, in vec2 fragCoord ) { fragColor = Mode2_Reproject_mainBufferB( fragCoord, iTime, iResolution, iFrame, iMouse , iChannel0, iChannel1); }
 #define BUFFERA void mainImage( out vec4 fragColor, in vec2 fragCoord ) { fragColor = Mode2_Reproject_mainBufferA( fragCoord, iTime, iResolution, iFrame, iMouse , iChannel0, iChannel1); }
 #define IMAGE \
 

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin


      vec4 ppd; \
      fragColor = Mode2_Reproject_mainImage( fragCoord, iTime, iResolution, iFrame, iMouse, iChannel0, iChannel1, ppd ); \
      fragColor.rgb = postProcess( fragColor.rgb, fragCoord / iResolution.xy, ppd ); \
      fragColor.a = 0.; /* remember we have the depth in alpha, png save will go weird so clear */ \

                return fragColor;
            }

            ENDCG
        }
    }
}

#endif

