
Shader "Skybox/DeathStar"
{
    Properties
    {
        _MainTex0 ("tex2D", 2D) = "white" {}
        _MainTex1 ("tex2D", 2D) = "white" {}
        _MainTex2 ("tex2D", 2D) = "white" {}
        _MainTex3 ("tex2D", 2D) = "white" {}
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

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex0; 
            uniform sampler2D _MainTex1; 
            uniform sampler2D _MainTex2; 
            uniform sampler2D _MainTex3; 
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

// Author: ocb
// Title: Hope 

/***********************************************************************
Trying another algorithm to generate city.
Here I am using what I may call "variable size voxels".
In this shader the voxels are 2d as horizontal grid
I am using 3 grids with 3 differents sizes (sizes are not multiples)
with a little shift.
So the interconnection of these grids generates infinite lower-sized squares.

The algorithm is in the 3 first functions.

The trace function goes from square to square.
The map return the height of the square
the getNextPlan function return the closest "square wall".


************************************************************************/

#define PI 3.141592653589793
#define PIdiv2 1.57079632679489
#define TwoPI 6.283185307179586
#define INFINI 1000000.
#define MAXSTEP 127
#define TOP_SURFACE 2.81

#define SKY 1
#define BLOC 2
#define WIN 3
#define SIDE 4


#define moonCtr float3(.12644925908247,.26123147706175,.908043331742229)   // float3(.10644925908247,.266123147706175,.958043331742229)
#define moonShad float3(-.633724250524478,.443606975367135,.633724250524478)
#define moonRefl float3(.477784395284944,.179169148231854,.8600119115129)   //float3(.477784395284944,.179169148231854,.8600119115129)

int hitObj = SKY;
float hitScale = 1.;

float Hsh(in float v) {                         
    return frac(sin(v) * 437585.);
}


float Hsh2(in float2 st) {                        
    return frac(sin(dot(st,float2(12.9898,8.233))) * 43758.5453123);
}

// thanks IQ
float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return lerp( b, a, h ) - k*h*(1.0-h);
}


/*
this function return the closest side of a squared defined by 3 overlapping grids.
The scales are .13, .21, .47 so they almost never match together generating smaller
squares between their intersections.
Not necessary, but here I add a shift (-.7) and (-.3) to shift the grids from the other.
From a position inside a square, the distance to the previous side is frac() and the distance
to the next side is 1-frac. Next and previous depend on the direction of the ray floattor
(here v).
So d (depending on the sign of v),represent 1 in case of 1-frac()
The whole is divided by the ray to obtain the t parameter to reach the side.

This is done for the 3 grids.
Then selecting the smallest one (closest) in both direction (xz)
and finally keeping the smallest one.

Normal is returned too.
*/

float4 getNextPlan(in float2 xz, in float2 v, in float t){
    float2 s = sign(v);
    float2 d = step(0.,s);
    float2 dtp = (d-frac(xz*.13))/.13/v;
    float2 dtq = (d-frac((xz-.7)*.21))/.21/v;
    float2 dtr = (d-frac((xz-.3)*.47))/.47/v;

    float2 dmin = min(min(dtp,dtq),dtr);
    float tmin = min(dmin.x, dmin.y);
    
    s *= -step(dmin,float2(tmin,tmin));
    
    return float4(float3(s.x,0.,s.y),tmin);
}


/*
map() function generates the 3 grids.
A random height is found for each square of each grid Hp, Hq, Hr
Pp Pq and Pr are the probability for a square to have a non-null/null value.

The final height of the sub square is a weighted sum of each of the 3 grid square.
*/

float map(in float2 xz)
{
    float2 p = floor(xz*.13)/.13;
    float2 q = floor((xz-.7)*.21)/.21;
    float2 r = floor((xz-.3)*.47)/.47;
    
    float Hp = Hsh2(p), Hq = Hsh2(q), Hr = Hsh2(r);
    float Pp = step(.6,Hp), Pq = step(.6,Hq), Pr = step(.5,Hr);
    
    float tex = 1.*Hp*Pp + .5*Hq*Pq +.3*Hr*Pr;    
    hitScale = Pp + 2.5*Pq + 5.*Pr;
    
    return tex;
    
}



/*
Trace() is the raytracing function.
First of all, find the entry point on the Top surface. Top surface represents the highest
value for the bulding. So we do not cross dozen of voxels when there is no chance
to hit the "floor" of a voxel (building).

When under the top surface the ray will navigate from a voxel to another using the 
getNextPlan() function.

p is the position reached.

we mesure the distance of p from the floor (map(p)) and from the closest "voxel wall" getNextPlan(p)
if map(p) is the closest, the ray hit the floor in this voxel, break.
if wall is the closest, go to the next voxel.

When entering a new voxel, check if below the floor.
if below the floor, so you hit a building wall, return wall info, break.

Not sure to be clear...
*/

float4 trace(in float3 pos, in float3 ray)
{
    float dh = 0.;
    float t = 0.;
    if(pos.y > TOP_SURFACE){
        if(ray.y >= 0.) return float4(float3(0.,0,0),INFINI);
        t = (TOP_SURFACE - pos.y)/ray.y + 0.00001;
    }
    
    float4 wall = float4(0.,0,0,0);
    
    for(int i = 0;i<MAXSTEP;i++){
        
        float3 p = pos+t*ray;
        if(p.y > TOP_SURFACE) break;
        
        float dh = p.y - map(p.xz);
        if(dh<.0) return float4(wall.xyz,t-.00001);
        
        wall = getNextPlan(p.xz,ray.xz,t);
        float tt = 0.;
        if(ray.y < 0.){
            float th = dh/(-ray.y);
            tt = min(wall.w,th);
            if(tt==th) return float4(0.,1.,0.,t+tt);
        }
        else tt = wall.w;
        
        t+= tt+.00001;
        if(t>250.) break;
    }
    
    
    return float4(0.,0.,0.,INFINI);
}


float4 boxImpact( in float3 pos, in float3 ray, in float3 ctr, in float3 dim) 
{
    float3 m = 1.0/ray;
    float3 n = m*(ctr-pos);
    float3 k = abs(m)*dim;
    
    float3 t1 = n - k;
    float3 t2 = n + k;

    float tmax = max( max( t1.x, t1.y ), t1.z );
    float tmin = min( min( t2.x, t2.y ), t2.z );
    
    if( tmax > tmin || tmin < 0.0) return float4(float3(0.,0,0),INFINI);

    float3 norm = -sign(ray)*step(t2, float3(tmin,tmin,tmin));
    return float4(norm, tmin);
    
}


bool checkWindow(in float3 ctr){
    float hash = Hsh2(ctr.xz+ctr.yy);
    float a = step(.3,hash)*step(((ctr.y)%10.),0.);
    float b = step(.6,hash)*step(((ctr.y-1.)%10.),0.);
    return bool(a+b);
}

float4 traceWindow(in float3 pos, in float3 ray, in float t, in float3 norm){
    float3 p = pos + t*ray;
    float4 info = float4(norm,t);

    float3 boxDim = float3(.25,.025,.25);
    float3 boxCtr;
    
    for(int i=0; i<5; i++){
        boxCtr = float3(floor(p.x*2.),floor(p.y*20.),floor(p.z*2.));
        if(checkWindow(boxCtr)){
            hitObj = SIDE;
            float tf = t + .1/dot(ray,-norm);
            info = boxImpact(pos, ray, (boxCtr+.5)*float3(.5,.05,.5), boxDim);
            if(tf < info.w){
                hitObj = WIN;
                info = float4(norm,tf);
                break;
            } 
            p = pos + (info.w+.001)*ray;
        }
        else break;
    }
    return info;
}

float3 moonGlow(in float3 ray){
    float a = dot(moonCtr, ray);
    float dl = dot(moonRefl,ray);
    float moon = smoothstep(.9,.902,a);
    float shad = 1.-smoothstep(.4,.7,dot(moonShad, ray));
    float refl = .7*smoothstep(.99,1.,dl);
    float clouds = min(1.,2.*tex2D(_MainTex1,float2(2.,-2.)*ray.xy-float2(.2,.3)).r);
    float3 col = .8*(float3(0.,.3,.6)+(1.-clouds))*moon+refl;
    col += float3(.3,.5,.8)*smoothstep(.88,.90,a)*(1.-smoothstep(.89,.95,a))*(dl-.9)*15.;
    col *= shad;
    col -= float3(.1,.3,.5)*(1.-moon*shad);
    col = clamp(col,0.,1.);
    return col;
}

float3 stars(in float3 ray){
    float3 col = float3(0.,0,0);
    float az = atan2(-.5*ray.x,.5*ray.z)/PIdiv2;
    float2 a = float2(az,ray.y);
    
    float gr = .5+a.x+a.y;
    float milky = 1.-smoothstep(0.,1.2,abs(gr));
    float nebu = 1.-smoothstep(0.,.7,abs(gr));

    float3 tex = tex2D(_MainTex3,a+.3).rgb;
    float3 tex2 = tex2D(_MainTex3,a*.1).rgb;
    float3 tex3 = tex2D(_MainTex3,a*5.).rgb;
    float dark = 1.-smoothstep(0.,.3*tex.r,abs(gr));
    
    float2 dty =a*12.;
    col += step(.85,Hsh2(floor(dty)))*(tex+float3(.0,.1,.1))*max(0.,(.01/length(frac(dty)-.5)-.05));
    
    dty =a*30.;
    col += step(.8,Hsh2(floor(dty)))*tex*max(0.,(.01/length(frac(dty)-.5)-.05))*milky;
    
    dty =a*1000.;
    col += max(0.,Hsh2(floor(dty))-.9)*3.*tex3*milky;
    
    col += (.075+.7*smoothstep(.1,1.,(tex+float3(.15,0.,0.))*.3))*nebu;
    col += .5*smoothstep(0.,1.,(tex2+float3(0.,.2,.2))*.2)*milky;
    col -= .15*(tex3 * dark);
    
    return col;
}

float3 fewStars(in float3 ray){
    float3 col = float3(0.,0,0);
    float az = atan2(-.5*ray.x,.5*ray.z)/PIdiv2;
    float2 a = float2(az,ray.y);
    
    float3 tex = tex2D(_MainTex3,a+.3).rgb;
    float2 dty =a*14.;
    col += step(.85,Hsh2(floor(dty)))*(tex+float3(.0,.1,.1))*max(0.,(.01/length(frac(dty)-.5)-.05));

    return col*(1.-smoothstep(.6,.9,dot(moonCtr,ray)));
}


bool shadTrace(in float3 pos, in float3 v){
    float dh = 0.;
    float t = 0.;
    float4 wall = float4(0.,0,0,0);
    
    for(int i = 0;i<10;i++){       
        float3 p = pos + t*v;
        if(p.y > TOP_SURFACE) break;       
        float dh = p.y - map(p.xz);
        if(dh<.0) return true;       
        wall = getNextPlan(p.xz,v.xz,t);       
        t+= wall.w + .0001 ;
    }   
    return false;   
}

float shadowfunc(in float3 p){
    p += .00001*moonRefl;
    if(shadTrace(p,moonRefl)) return .2;
    else return 1.;
}

float3 winGlow(in float2 uv){
    uv.x *= .2;
    uv.y *= .5;
    float2 k1 = (uv-.05*sin(uv*10.))*10.,
         k2 = (uv-.02*sin(uv*25.))*25.,
         k3 = (uv-.01*sin(uv*50.))*50.;
    
    
    float2 p = floor(k1)/10.,
         q = floor(k2)/25.,
         s = floor(k3)/50.;
    
    float2 bp = abs(frac(k1)-.5)
            + abs(frac(k2)-.5)
            + abs(frac(k3)-.5);
    bp /= 1.5;
    bp*=bp*bp;
    
    float3 tex = tex2D(_MainTex2,p).rgb
             + tex2D(_MainTex2,q).rgb
             + tex2D(_MainTex2,s).rgb;
    
    tex += .5*(bp.x+bp.y);
    tex *= smoothstep(1.,2.8,tex.r);
    
    return tex;
}


float metalPlate(in float2 st){
    float coef = 0.;
    
    float2 p = floor(st);
    float hp = Hsh2(p*0.543234); hp *= step(.2,abs(hp-.5));
    float2 fp = frac(st)-.5;
    float2 sfp = smoothstep(.475,.5,abs(fp));
    
    st *= float2(.5,1.);
    float2 q = floor(st*4.-.25);
    float hq = Hsh2(q*0.890976); hq *= step(.35,abs(hq-.5));
    float2 fq = frac(st*4.-.25)-.5;
    float2 sfq = smoothstep(.45,.5,abs(fq));
    
    st *= float2(5.,.1);
    float2 r = floor(st*8.-.25);
    float hr = Hsh2(r*0.123456); hr *= step(.47,abs(hr-.5));
    float2 fr = frac(st*8.-.25)-.5;
    float2 sfr = smoothstep(.4,.5,abs(fr));
    
    float h = max(max(hp,hq),hr);
    if(bool(h)){
        float2 plate =    step(h,hp)*sfp*sign(fp)
                      + step(h,hq)*sfq*sign(fq) 
                      + step(h,hr)*sfr*sign(fr);
        
        coef += .2*h+.8;
        coef += .5*min(1.,plate.x+plate.y);
    }
    else coef = 1.;
    
    return coef;
}


float lightPath(in float2 uv){    
    return step(.965,Hsh(floor(uv.x*10.)))+step(.965,Hsh(floor(uv.y*10.)));
}

float3 groundLight(in float3 pos, in float3 ray, in float t){
    float3 col = float3(0.,0,0);
    float ty = (.00001-pos.y)/ray.y;
    ty += step(ty,0.)*INFINI;
    pos += ty*ray;
    if(ty<t) col += (.05/length(pos.xz*20. - float2(floor(pos.xz*20.)+.5))-.08)
                    * lightPath(pos.xz);
    return col;
}


float flare(in float3 s, in float3 ctr){
    float c = 0.;
    s = normalize(s);
    float sc = dot(s,-moonRefl);
    c += .5*smoothstep(.99,1.,sc);
    
    s = normalize(s+.9*ctr);
    sc = dot(s,-moonRefl);
    c += .3*smoothstep(.9,.91,sc);
    
    s = normalize(s-.6*ctr);
    sc = dot(s,-moonRefl);
    c += smoothstep(.99,1.,sc);
    
    return c;
}


float3 lensflare3D(in float3 ray, in float3 ctr)
{
    float3 red = float3(1.,.6,.3);
    float3 green = float3(.3,1.,.6);
    float3 blue = float3(.6,.3,1.);
    float3 col = float3(0.,0,0);
    float3 ref = reflect(ray,ctr);

    col += red*flare(ref,ctr);
    col += green*flare(ref-.15*ctr,ctr);
    col += blue*flare(ref-.3*ctr,ctr);
    
    ref = reflect(ctr,ray);
    col += red*flare(ref,ctr);
    col += green*flare(ref+.15*ctr,ctr);
    col += blue*flare(ref+.3*ctr,ctr);
    
    float d = dot(ctr,moonRefl);
    return .4*col*max(0.,d*d*d*d*d);
}



float3 getCamPos(in float3 camTarget){
    float   rau = 15.,
            alpha = 0,  //iMouse.x/iResolution.x*4.*PI,
            theta = 0;  //iMouse.y/iResolution.y*PI+(PI/2.0001);  
            
            // to start shader
//            if (iMouse.xy == float2(0.)){
//                float ts = smoothstep(18.,22.,_Time.y)*(_Time.y-20.);
//                float tc = smin( _Time.y, 30., 3. );
//                alpha = -2.-ts*.05;
//                theta = 1.5-tc*.05;
//            }
    return rau*float3(-cos(theta)*sin(alpha),sin(theta),cos(theta)*cos(alpha))+camTarget;
}

float3 getRay(in float2 st, in float3 pos, in float3 camTarget){
    float   focal = 1.;
    float3 ww = normalize( camTarget - pos);
    float3 uu = normalize( cross(ww,float3(0.0,1.0,0.0)) ) ;
    float3 vv = cross(uu,ww);
    // create view ray
    return normalize( st.x*uu + st.y*vv + focal*ww );
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex0, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

    float2 st = ( fragCoord.xy
               // here below, shifting a little even coordinates (H and V).
               // Do not really manage aliasing, but reduce strongly flickering due to aliasing
               // Far from a good job, but it is absolutly costless!!
               // comment the line below to compare
               -.5*float2(((fragCoord.y)%2.),((fragCoord.x)%2.))
               - .5*float2(800,450) ) / 450; //               - .5*iResolution.xy ) / iResolution.y; 
    float ti = _Time.y*.02;
    
    // camera def
    float3 camTarget = float3(-50.*sin(2.*ti),2.1,-30.*cos(3.*ti));
    //float3 camTarget = float3(0.);
       
    float3 pos = ro; //getCamPos(camTarget);
    pos.y = max(pos.y,map(pos.xz)+.1);
    
    float3 ray = rd; //getRay(st, pos,camTarget);
    
    bool moonside = bool(step(0.,dot(ray,moonRefl)));
    
    float3 color = float3(.0,0,0);
    float t = 0.;
    float3 norm = float3(0.,0,0);

    float4 info = trace(pos, ray);
    float sc = hitScale;
    t = info.w;
    norm = info.xyz;
    
    float shadow = shadowfunc(pos+t*ray);
    
    if(t==INFINI){
        if(moonside){
            color += moonGlow(ray);
            color += fewStars(ray);
        }
        else color += stars(ray);
    }
    else{
        if(!bool(norm.y)) {
            info = traceWindow(pos ,ray, t, norm);
            if(bool(info.w)) {
                norm = info.xyz;
                t = info.w;
            }
        }
        
        float3 p = pos + t*ray;

        if(hitObj == WIN){
            float3 window = winGlow( ((p.xy+p.z)*norm.z + (p.zy+p.x)*norm.x))*(1.-norm.y);
            float3 refl = reflect(ray,norm);
            color += smoothstep(.95,1.,dot(moonRefl,refl))*norm.z*step(1.,shadow);
            color += window*min(1., 30./t);
        }
        
        else{
            float2 side = .1*p.xz*norm.y + .5*p.xy*norm.z + .5*p.zy*norm.x;
            color += tex2D(_MainTex0,side).rgb;
            color *= metalPlate(4.*side);
            color += .015*float3(.5*sc,abs(sc-4.),8.-sc) * min(1.,10./t);

            color *= clamp(dot(norm, moonRefl)+.2,.3,1.);
            if(hitObj == SIDE) color += float3(.1,.05,.0);
            else color *= shadow;
            
            float3 refl = reflect(ray,norm);
            color += .3*smoothstep(.9,1.,dot(moonRefl,refl))*norm.z*step(1.,shadow);;

            color += tex2D(_MainTex2,p.xz*.1).rgb*groundLight(pos, ray, t);
            color -= 2.*tex2D(_MainTex2,p.xz*.2).rgb*(norm.x+norm.z)*lightPath(p.xz)*step(.001,p.y)*step(p.y,.08);
            color = clamp(color,0.,1.);
        }
        color *= min(1., 80./t);
    }
     if(moonside)
        if(!shadTrace(pos,moonRefl))
            color += lensflare3D(ray, getRay(float2(0.,0), pos,camTarget));
    
    fragColor = float4(color,1.);

                return fragColor;
            }

            ENDCG
        }
    }
}
