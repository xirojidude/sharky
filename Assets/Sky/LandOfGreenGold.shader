
Shader "Skybox/LandOfGreenGold"
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

//////////////////////////////////////////////////////
// Land of Green Gold by Timo Kinnunen 2017
//
// Based on Fur Trees by eiffie
// @ https://www.shadertoy.com/view/lts3zr
//
// Based on Faster Voronoi Edge Distance by tomkh
// @ https://www.shadertoy.com/view/llG3zy
//
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

#define NO_DEBUG

// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// by Tomasz Dobrowolski' 2016

// Based on https://www.shadertoy.com/view/ldl3W8 by Inigo Quilez
// And his article: http://www.iquilezles.org/www/articles/voronoilines/voronoilines.htm

// This is based on Inigo Quilez's distance to egdes,
// except I consider here faster variants:
// * 3x3 scan at the cost of small inaccuracies
// * 4x4 scan in the second pass that has no accuracy-loss to IQ's version
// * 4x4 in both passes that is more accurate than original IQ's version
//   (and still has less iterations 32=4*4*2 vs 34=3*3+5*5)

// Original IQ's algorithm is flawed (mathematically incorrect)
// i.e. for all possible hash functions, as in this counter-example:
// https://www.shadertoy.com/view/4lKGRG

// Basically in the original IQ's implementation,
// he was storing closest cell "mg" in the first pass
// and using it for the second pass.
// If we want 3x3 scan in the second pass it is enough to continue search
// from the same (current fragment) cell and limit search space
// to only neighbouring cells (thus "mg" can be ignored).
// In fact, searching around "mg" makes it worse (see my illustration below).
// For 4x4 variant we have to set the center of search 
// based on which half of the current fragment cell we are in.
// Note: 
//   The second pass scan area has nothing to do with the position
//   of the closest point.
//   Here is an illustration of my improved algorithm:
//   http://ricedit.com/second_order_voronoi_03.png

// Pick approximation level:
//   0 = 3x3 scan for both passes (occasional issues, but the fastest)
//   1 = 3x3 + 4x4 scan (good in most cases, if every cell has diameter < 1)
//   2 = 4x4 scan for both passes (improved accuracy)
//   3 = 3x3 + 5x5 scan (original IQ's)
#define SECOND_PASS 0

//#define ANIMATE

// How far cells can go off center during animation (must be <= .5)
#define ANIMATE_D .5

// Points cannot be closer than sqrt(EPSILON)
#define EPSILON .00001

// How freely are cell centers placed (must be <= 1.)
#define PLACE_D .875

vec2 hash2(vec2 p)
{
    #if 1
       // Dave Hoskin's hash as in https://www.shadertoy.com/view/4djSRW
       vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
       p3 += dot(p3, p3.yzx+19.19);
       vec2 o = PLACE_D*(fract(vec2((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y))-.5)+.5;
    #else
       // Texture-based
       vec2 o = texture( iChannel0, (p+0.5)/256.0, -100.0 ).xy;
    #endif
    #ifdef ANIMATE
       o = 0.5 + ANIMATE_D*sin( iTime + o*6.2831853 );
    #endif
   return o;
}
struct VoronoiData {
    float md;
    vec2 mr;
    vec2 mi;
};
#if SECOND_PASS == 0
//---------------------------------------------------------------
// Fastest version with 3x3 scan in the second pass
//---------------------------------------------------------------
VoronoiData voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
    vec2 mr;
    vec2 mi;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mi = n + g;
        }
    }

    //----------------------------------
    // second pass: distance to borders,
    // visits only neighbouring cells
    //----------------------------------
    md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) {// skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
        }
    }

    VoronoiData VD;
    VD.md = md;
    VD.mr = mr;
    VD.mi = mi;
    return VD; //return VoronoiData( md, mr, mi );
}


#elif SECOND_PASS == 1
//---------------------------------------------------------------
// Approximation with 4x4 scan in the second pass
// Good enough in most cases
//---------------------------------------------------------------

VoronoiData voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
    vec2 mr;
    vec2 mi;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mi = n + g;
        }
    }
    
    // Set center of search based on which half of the cell we are in,
    // since 4x4 is not centered around "n".
    vec2 mg = step(.5,f) - 1.;

    //----------------------------------
    // second pass: distance to borders,
    // visits two neighbours to the right/down
    //----------------------------------
    md = 8.0;
    for( int j=-1; j<=2; j++ )
    for( int i=-1; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return VoronoiData( md, mr, mi );
}

#elif SECOND_PASS == 2
//---------------------------------------------------------------
// 4x4 scan in both passes = most accurate
//---------------------------------------------------------------

VoronoiData voronoi( in vec2 x )
{
#if 1
    // slower, but better handles big numbers
    vec2 n = floor(x);
    vec2 f = fract(x);
    vec2 h = step(.5,f) - 2.;
    n += h; f -= h;
#else
    vec2 n = floor(x - 1.5);
    vec2 f = x - n;
#endif

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
    vec2 mr;
    vec2 mi;

    float md = 8.0;
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mi = n + g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return VoronoiData( md, mr, mi );
}

#else
//---------------------------------------------------------------
// Slowest with 5x5 scan in the second pass
// Original Implementation by Inigo Quilez 
// as in https://www.shadertoy.com/view/ldl3W8
//---------------------------------------------------------------

VoronoiData voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
    vec2 mg, mr;
    vec2 mi;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
            mi = n + g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return VoronoiData( md, mr, mi );
}

//---------------------------------------------------------------
#endif

vec3 plot( vec2 p, float ss )
{
    VoronoiData c = voronoi( p );
    
    // Colors:
    vec3 interior = vec3(.3,.2,.2);
    vec3 border = vec3(.5,.6,0);
    vec3 point0 = vec3(.025,.025,.0125);
    
    float d = length(c.mr);
    vec3 col =
        mix(
            mix(
                interior*(.63-1.1*c.md),
                border*.2+.25*c.md,
                smoothstep(.22,.04,c.md)
            ),
            point0,
            smoothstep(.427,.05,d)
        );
    
    return col;
}

void mainImage00( out vec4 fragColor, in vec2 fragCoord )
{
    float sc = step(512., iResolution.y)*4. + 4.; // scale differently for fullscreen
    float ss = sc / iResolution.y; // size of 1 pixel
    vec2 uv = (fragCoord.xy - iResolution.xy*.5) * ss;
    fragColor = vec4(plot(uv, ss), 1.);
}




//attempting some distance estimated fur and shading with one extra DE calc
#define AUTO_OVERSTEP

#define time iTime
#define size iResolution

#define TAO 6.283
vec2 rotate(vec2 v, float angle) {return cos(angle)*v+sin(angle)*vec2(v.y,-v.x);}
vec2 kaleido(vec2 v, float power){return rotate(v,floor(.5+atan(v.x,-v.y)*power/TAO)*TAO/power);}

vec2 kaleido6(vec2 v){return rotate(v,floor(0.5+atan(v.x,-v.y)*0.95493)*1.0472);}
vec2 kaleido12(vec2 v){return rotate(v,floor(0.5+atan(v.x,-v.y)*1.90986)*0.5236);}
float rndStart(vec2 co){return 0.1+0.9*fract(sin(dot(co,vec2(123.42,117.853)))*412.453);}

vec3 mcol;//material color
mat2 r45=mat2(0.7071,0.7071,-0.7071,0.7071);
mat2 r30=mat2(0.866,0.5,-0.5,0.866);
mat2 rtrn=mat2(0.9689,-0.2474,0.2474,0.9689);

VoronoiData theVoro;
float DE(in vec3 z0){
    VoronoiData voro = voronoi( z0.xz );
    theVoro = voro;
    vec2 id = voro.mi;
    z0.xz = voro.mr;

    float cyl=length(z0.xz);
    float d=100.0,dt=cyl+z0.y*0.025;
    
    for(int i=0;i<2;i++){
        vec3 z=z0;
        z.xz=rotate(z.xz,id.x*2.+id.y*3.+float(i*2));
        z.y+=float(i)*(0.125+.0625*mod(id.x-id.y,2.0));
        float c=floor(z.y*4.0);
        //z.yz=rotate(z.yz,-z.z*0.79*(1.0+c*0.1));
        float bm=-z.y-2.0+cyl*0.01;
        z.y=mod(z.y,0.25)-0.05;
        if(i<=1)mul(z.xz=z.xz,rtrn)*float(i+i-1);
        z.xz=kaleido(z.xz,2.0-c);
        z.yz=mul(rtrn,z.yz);
        bm=max(bm,-z.z+c*0.086);//0.065);
        dt=min(dt,max(max(abs(z.x),abs(z.y)),bm))-0.001-z.z*0.003;
        float c2=floor(z.z*16.0);
        z.z=mod(z.z,0.0625)-0.045;//0.049;
        z.xy=rotate(z.xy,c2*0.25);
        z.xy=kaleido12(z.xy);
        z.yz=mul(z.yz,r30);
        d=min(d,max(max(max(abs(z.x),abs(z.z)),-z.y-0.05+c*0.005),bm));
    }
    if(dt<d){
        d=dt;
        mcol=vec3(0.5,0.1,0.0);
    }else{
        mcol=vec3(0.5,0.6,0.2);
        //mcol*=1.0+(-z0.y*0.5975)*abs(z0.x+z0.z)/cyl;
    }
    mcol*=cyl + 0.5 + z0.y*0.5;//kind of what iq suggested

    return max(0.0,max(d,max(z0.y,-z0.y-2.0)));
}

mat3 lookat(vec3 fw,vec3 up){
    fw=normalize(fw);vec3 rt=normalize(cross(fw,up));return mat3(rt,cross(rt,fw),fw);
}
float fog(float t){return exp(-t*0.02);}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

    float zoom=2.0;
    float px=2.0/(size.y*zoom);//find the pixel size
    float ipx = 1.0/px;
    float tim=time;

    #ifndef NO_DEBUG
    float mark0 = fragCoord.x-iResolution.x*.5;
    float mark1 = abs(mark0)-iResolution.x*.1875;
    float mark2 = mark1-iResolution.x*.0625;
    float mark3 = mark1-iResolution.x*.1250;
    float mark4 = mark1-iResolution.x*.1875;
    float mark5 = mark1-iResolution.x*.2500;
    tim += mark0>0.?30.:0.;
    #endif

    tim *= 0.3;
    //position camera
//    vec3 ro=vec3(0.5*sin(tim*0.43),18.*(sin(tim*.4)+.975),tim*20.);
//    vec3 rd=normalize(vec3((2.0*fragCoord.xy-size.xy)/size.y,zoom));
//    rd=lookat(vec3(-1.15*4.+8.*0.75*sin(tim),1.-1.5*sin(tim*.02),tim*20.+2.0)-ro,vec3(0.0,1.0,0.0))*rd;
    //ro=eye;rd=normalize(dir);
    vec3 ld=normalize(vec3(-0.4,0.75,-0.4));//direction to light
    vec3 bcol=1.0-clamp(vec3(rd.y,rd.y+0.1*exp(-abs(rd.y*15.0)),0.5*exp(-abs(rd.y*5.0))),0.,1.);//backcolor
    //march
    
    fragColor=vec4(0,0,0,1);
    
    float tG=abs((-2.0-ro.y)/rd.y);
    float d;
    float pd=10.0;
    float os=0.0;
    float steps=0.0;
    vec2 g=ro.xz+rd.xz*tG;
    float noise = rndStart(fragCoord.xy);
    const float rndBands = 1.;
    float t=floor(rndBands*DE(ro)*noise+.5)/rndBands;
    float MIN_DIST=px*0.1;
    vec3 pos;
    vec4 col=vec4(0.0,0,0,0);//color accumulator
    for(int i=0;i<78;i++){
        pos = ro+rd*t;
        d=DE(pos);
        float d1=max(d,px*t*0.5);
#ifdef AUTO_OVERSTEP
        if(d1>os){      //we have NOT stepped over anything
            if(t>tG)break;
            os=0.28*d1*d1/pd;//calc overstep based on ratio of this step to last
            steps=d1+os;    //add in the overstep
            pd=d1;  //save this step length for next calc
        }else{
            steps=-os;d1=1.0;pd=10.0;os=0.0;//remove ALL of overstep
        }
#else
            steps=d1;
#endif
        if(d1<px*t){
            vec3 scol=mix(mcol,bcol,clamp(min(t*0.05,1.0),0.,1.));
            float d2=DE(pos+ld*px*t);
            float shad=1.4-0.5*clamp((d2/d)+1.,0.,10.);
            scol=scol*shad+vec3(0.02,0.05, -0.125)*(shad+1.5);
            vec2 q = mod(pos.xz,2.0)-vec2(1.0,1);
            scol *= clamp(length(q) + 0.8 + pos.y*0.5,0.,1.);
            float alpha=(1.0-col.w)*clamp(1.0-d1/(px*t),0.,1.);
            //alpha=1.;
            col+=vec4(clamp(scol,0.0,1.0),1.0)*alpha;
            if(col.w>0.99)break;
        }
        t+=steps;
    }
    

    float treeMask = col.a;
    float horizon = .5+.5*cos(radians(clamp(rd.y*12.*180.,-180.,180.)));
    float attenFar = clamp(log(max(1.0,(1./30.)*mix(t,t-25.,horizon)))*2.5,0.,1.);//smoothstep(50.0,75.0,t);
    float distantFog = (1.0-0.292*attenFar);
    float onlyFog = fog(abs(t));
    float fogNear=1.92-.325*col.w-col.w*onlyFog;
    float farFog = 1.0-fogNear;
    float treeMaskFar = 1.0-treeMask*clamp(fogNear*10.,0.,1.);
    float horizonFog = pow(horizon*.99,16.)*treeMaskFar;
    float groundCover = fogNear*(1.0-col.a)*distantFog;
    vec4 groundCol = vec4(0,0,0,clamp(-64.0*rd.y,0.,1.));
    //color the ground 
    if(groundCol.a>0.0){
        vec3 gcol = plot(g,px);
        ro+=rd*tG;
        float s=1.0,dst=0.1;
        t=DE(ro)*rndStart(fragCoord.xy);
        for(int i=0;i<4;i++){
            float d=max(0.0,DE(ro+ld*t)*1.5)+0.05;
            s=min(s,3.0*d/t);
            t+=dst;
            dst*=2.0;
        }
        gcol*=0.2+0.8*s;
        //col.rgb+=clamp(gcol*fogNear*(1.0-col.a)*(1.0-0.292*attenFar),0.,1.);
        //col.a+=(1.0-col.a)*clamp(fogNear*(1.0-col.a)*(1.0-0.292*attenFar),0.,1.);
        groundCol.rgb+= clamp(gcol*groundCover,0.,1.);
        groundCol.a+=(1.0-col.a)*clamp(groundCover,0.,1.);
        //groundCol.rgb = clamp(gcol*fogNear*(1.0-col.a)*(1.0-0.292*attenFar),0.,1.);
    }
    float groundMask = (1.-col.a);
    col+=groundCol*groundMask;
    float skyMask = (1.-col.a);
    //col.a+=(1.0-col.a)*clamp(fogNear*(1.0-col.a)*(1.0-0.292*attenFar),0.,1.);
    col.rgb += clamp(bcol*skyMask,0.,1.);
    //col.rgb = mix(col.rgb,bcol,attenFar);
    //col.rgb = mix(col.rgb,bcol,attenFar*(1.0-col.a));
    col.rgb = mix(col.rgb,bcol,horizonFog);
    col.rgb += bcol*horizon*.5;
    //col.rgb = mix(col.rgb,bcol,attenFar*(1.0-col.a)*(1.0-horizonFog)+horizonFog);
    float noSkyMask = clamp(1.-onlyFog,0.,1.);
    float groundedFog = clamp((fogNear-.8)*(treeMask+horizon*.5)*1.75,0.,1.)*.25;
    col.rgb += bcol*(groundedFog);
    
    fragColor=vec4(col.rgb,1.0);
    #ifndef NO_DEBUG
    fragColor.rgb=mark1>0.?vec3(0,0,1)*vec3(groundCover,groundMask,treeMaskFar):fragColor.rgb;
    fragColor.rgb=mark2>0.?vec3(1,1,1)*vec3(attenFar,distantFog,farFog):fragColor.rgb;
    fragColor.rgb=mark3>0.?vec3(0,1,0)*vec3(onlyFog,fogNear,treeMask):fragColor.rgb;
    fragColor.rgb=mark4>0.?vec3(1,1,1)*vec3(horizon,horizonFog,skyMask):fragColor.rgb;
    fragColor.rgb=mark5>0.?vec3(0,1,0)*vec3(noSkyMask,groundedFog,skyMask):fragColor.rgb;
    #endif


                return fragColor;
            }

            ENDCG
        }
    }
}
