
Shader "Skybox/Canyon"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _MainTex1 ("Texture", 2D) = "white" {}
        _MainTex2 ("Texture", 2D) = "white" {}
        _MainTex3 ("Texture", 2D) = "white" {}
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
            #define dFdx ddx
            #define dFdy ddy
            #define _iChannel0 _MainTex

            #define textureLod(a,b,c) tex2Dlod(a,float4(b,0,c))
            #define atan(a,b) atan2(b,a)

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex,_MainTex1,_MainTex2,_MainTex3; 
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



// Created by inigo quilez - iq/2014
// I share this piece (art and code) here in Shadertoy and through its Public API, only for educational purposes. 
// You cannot use, sell, share or host this piece or modifications of it as part of your own commercial or non-commercial product, website or project.
// You can share a link to it or an unmodified screenshot of it provided you attribute "by Inigo Quilez, @iquilezles and iquilezles.org". 
// If you are a techer, lecturer, educator or similar and these conditions are too restrictive for your needs, please contact me and we'll work it out.

//-----------------------------------------------------------------------------------

#define LOWDETAIL
//#define HIGH_QUALITY_NOISE

float noise2( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
#ifndef HIGH_QUALITY_NOISE
    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
    vec2 rg = textureLod( _MainTex2, (uv+ 0.5)/256.0, 0.0 ).yx;
#else
    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);
    vec2 rg1 = textureLod( _MainTex2, (uv+ vec2(0.5,0.5))/256.0, 0.0 ).yx;
    vec2 rg2 = textureLod( _MainTex2, (uv+ vec2(1.5,0.5))/256.0, 0.0 ).yx;
    vec2 rg3 = textureLod( _MainTex2, (uv+ vec2(0.5,1.5))/256.0, 0.0 ).yx;
    vec2 rg4 = textureLod( _MainTex2, (uv+ vec2(1.5,1.5))/256.0, 0.0 ).yx;
    vec2 rg = mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
#endif  
    return mix( rg.x, rg.y, f.z );
}


float hash(float3 v)
{
    return frac(sin(dot(v, float3(11.51721, 67.12511, 9.7561))) * 1551.4172);   
}

float noise1(float3 v)
{
    float3 rootV = floor(v);
    float3 f = smoothstep(0.0, 1.0, frac(v));
    
    //Cube vertices values
    float n000 = hash(rootV);
    float n001 = hash(rootV + float3(0,0,1));
    float n010 = hash(rootV + float3(0,1,0));
    float n011 = hash(rootV + float3(0,1,1));
    float n100 = hash(rootV + float3(1,0,0));
    float n101 = hash(rootV + float3(1,0,1));
    float n110 = hash(rootV + float3(1,1,0));
    float n111 = hash(rootV + float3(1,1,1));
    
    //trilinear interpolation
    float4 n = lerp(float4(n000, n010, n100, n110), float4(n001, n011, n101, n111), f.z);
    n.xy = lerp(float2(n.x, n.z), float2(n.y, n.w), f.y);
    return lerp(n.x, n.y, f.x);
}




// Standard 2x2 hash algorithm.
float2 hash22(float2 p) {
//    return float2(0,0);
    // Faster, but probaly doesn't disperse things as nicely as other methods.
    float n = sin(dot(p, float2(113, 1)));
    p = frac(float2(2097152, 262144)*n)*2. - 1.;
    #ifdef RIGID
    return p;
    #else
    return cos(p*6.283 + _Time.y);
    //return abs(frac(p+ _Timw.y*.25)-.5)*2. - .5; // Snooker.
    //return abs(cos(p*6.283 + _Time.y))*.5; // Bounce.
    #endif

}

 // Gradient noise. Ken Perlin came up with it, or a version of it. Either way, this is
// based on IQ's implementation. It's a pretty simple process: Break space into squares, 
// attach random 2D floattors to each of the square's four vertices, then smoothly 
// interpolate the space between them.
float n2D(in float2 f){
//return 0;    
    // Used as shorthand to write things like float3(1, 0, 1) in the short form, e.yxy. 
   const float2 e = float2(0, 1);
   
    // Set up the cubic grid.
    // Integer value - unique to each cube, and used as an ID to generate random floattors for the
    // cube vertiies. Note that vertices shared among the cubes have the save random floattors attributed
    // to them.
    float2 p = floor(f);
    f -= p; // fracional position within the cube.
    

    // Smoothing - for smooth interpolation. Use the last line see the difference.
    //float2 w = f*f*f*(f*(f*6.-15.)+10.); // Quintic smoothing. Slower and more squarish, but derivatives are smooth too.
    float2 w = f*f*(3. - 2.*f); // Cubic smoothing. 
    //float2 w = f*f*f; w = ( 7. + (w - 7. ) * f ) * w; // Super smooth, but less practical.
    //float2 w = .5 - .5*cos(f*3.14159); // Cosinusoidal smoothing.
    //float2 w = f; // No smoothing. Gives a blocky appearance.
    
    // Smoothly interpolating between the four verticies of the square. Due to the shared vertices between
    // grid squares, the result is blending of random values throughout the 2D space. By the way, the "dot" 
    // operation makes most sense visually, but isn't the only metric possible.
    float c = lerp(lerp(dot(hash22(p + e.xx), f - e.xx), dot(hash22(p + e.yx), f - e.yx), w.x),
                  lerp(dot(hash22(p + e.xy), f - e.xy), dot(hash22(p + e.yy), f - e.yy), w.x), w.y);
    
    // Taking the final result, and converting it to the zero to one range.
    return c*.5 + .5; // Range: [0, 1].
}

//-----------------------------------------------------------------------------------
const mat3 m = mat3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
                    -0.60, -0.48,  0.64 );

float displacement( vec3 p )
{
    float f;
    f  = 0.5000*noise1( p ); p = mul(m,p*2.02);
    f += 0.2500*noise1( p ); p = mul(m,p*2.03);
    f += 0.1250*noise1( p ); p = mul(m,p*2.01);
    #ifndef LOWDETAIL
    f += 0.0625*noise1( p ); 
    #endif
    return f;
}

vec4 texcube( sampler2D sam, in vec3 p, in vec3 n )
{
    vec4 x = texture( sam, p.yz );
    vec4 y = texture( sam, p.zx );
    vec4 z = texture( sam, p.xy );
    return (x*abs(n.x) + y*abs(n.y) + z*abs(n.z))/(abs(n.x)+abs(n.y)+abs(n.z));
}

vec4 textureGood( sampler2D sam, vec2 uv, float lo )
{
    uv = uv*1024.0 - 0.5;
    vec2 iuv = floor(uv);
    vec2 f = fract(uv);
    vec4 rg1 = textureLod( sam, (iuv+ vec2(0.5,0.5))/1024.0, lo );
    vec4 rg2 = textureLod( sam, (iuv+ vec2(1.5,0.5))/1024.0, lo );
    vec4 rg3 = textureLod( sam, (iuv+ vec2(0.5,1.5))/1024.0, lo );
    vec4 rg4 = textureLod( sam, (iuv+ vec2(1.5,1.5))/1024.0, lo );
    return mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
}

//-----------------------------------------------------------------------------------

float terrain( in vec2 q )
{
    float th = smoothstep( 0.0, 0.7, textureLod( _MainTex, 0.001*q, 0.0 ).x );
    float rr = smoothstep( 0.1, 0.5, textureLod( _MainTex1, 2.0*0.03*q, 0.0 ).y );
    float h = 1.9;
    #ifndef LOWDETAIL
    h += -0.15 + (1.0-0.6*rr)*(1.5-1.0*th) * 0.3*(1.0-textureLod( _MainTex, 0.04*q*vec2(1.2,0.5), 0.0 ).x);
    #endif
    h += th*7.0;
    h += 0.3*rr;
    return -h;
}

float terrain2( in vec2 q )
{
    float th = smoothstep( 0.0, 0.7, textureGood( _MainTex, 0.001*q, 0.0 ).x );
    float rr = smoothstep( 0.1, 0.5, textureGood( _MainTex1, 2.0*0.03*q, 0.0 ).y );
    float h = 1.9;
    h += th*7.0;
    return -h;
}

vec4 map( in vec3 p )
{
    float h = terrain( p.xz );
    float dis = displacement( 0.25*p*vec3(1.0,4.0,1.0) );
    dis *= 3.0;
    return vec4( (dis + p.y-h)*0.25, p.x, h, 0.0 );
}

vec4 raycast( in vec3 ro, in vec3 rd, in float tmax )
{
    float t = 0.1;
    vec3 res = vec3(0.0,0,0);
    for( int i=0; i<64; i++ )  //i<256; i++ )
    {
        vec4 tmp = map( ro+rd*t );
        res = tmp.ywz;
        t += tmp.x;
        if( tmp.x<(0.001*t) || t>tmax ) break;
    }

    return vec4( t, res );
}

vec3 calcNormal( in vec3 pos, in float t )
{
    vec2 eps = vec2( 0.005*t, 0.0 );
    return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<10; i++ )  //i<50; i++ )
    {
        float h = map(ro + rd*t).x;
        res = min( res, k*h/t );
        t += clamp( h, 0.5, 1.0 );
        if( h<0.001 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Oren-Nayar
float diffuse( in vec3 l, in vec3 n, in vec3 v, float r )
{
    float r2 = r*r;
    float a = 1.0 - 0.5*(r2/(r2+0.57));
    float b = 0.45*(r2/(r2+0.09));
    float nl = dot(n, l);
    float nv = dot(n, v);
    float ga = dot(v-n*nv,n-n*nl);
    return max(0.0,nl) * (a + b*max(0.0,ga) * sqrt((1.0-nv*nv)*(1.0-nl*nl)) / max(nl, nv));
}

vec3 cpath( float t )
{
    vec3 pos = vec3( 0.0, 0.0, 95.0 + t );
    
    float a = smoothstep(5.0,20.0,t);
    pos.xz += a*150.0 * cos( vec2(5.0,6.0) + 1.0*0.01*t );
    pos.xz -= a*150.0 * cos( vec2(5.0,6.0) );
    pos.xz += a* 50.0 * cos( vec2(0.0,3.5) + 6.0*0.01*t );
    pos.xz -= a* 50.0 * cos( vec2(0.0,3.5) );

    return pos;
}

mat3 setCamera( in vec3 ro, in vec3 ta, float cr )
{
    vec3 cw = normalize(ta-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.01+ _XYZPos;                                             // ray origin

    vec2 q = fragCoord.xy / iResolution.xy;
    vec2 p = -1.0 + 2.0 * q;
    p.x *= iResolution.x/iResolution.y;
    vec2 m = vec2(0.0,0);
//    if( iMouse.z>0.0 ) m = iMouse.xy/iResolution.xy;

    //-----------------------------------------------------
    // camera
    //-----------------------------------------------------

    float an = 0.5*(iTime-5.0);// + 12.0*(m.x-0.5);
//    vec3 ro = cpath( an + 0.0 );
    vec3 ta = cpath( an + 10.0 *1.0);
    ta = mix( ro + vec3(0.0,0.0,1.0), ta, smoothstep(5.0,25.0,an) );
//    ro.y = terrain2( ro.xz ) - 0.5;
    ta.y = ro.y - 0.1;
    ta.xy += step(0.01,m.x)*(m.xy-0.5)*4.0*vec2(-1.0,1.0);
    float rl = -0.1*cos(0.05*6.2831*an);
    // camera to world transform    
    mat3 cam = setCamera( ro, ta, rl );
    
    // ray
 //   vec3 rd = normalize( cam * vec3(p.xy, 2.0) );

    //-----------------------------------------------------
    // render
    //-----------------------------------------------------

    const vec3 klig = normalize(_SunDir); //normalize(vec3(-1.0,0.19,0.4));
    
    float sun = clamp(dot(klig,rd),0.0,1.0 );

    vec3 hor = mix( 1.2*vec3(0.70,1.0,1.0), vec3(1.5,0.5,0.05), 0.25+0.75*sun );
    
    vec3 col = mix( vec3(0.2,0.6,.9), hor, exp(-(4.0+2.0*(1.0-sun))*max(0.0,rd.y-0.1)) );
    col *= 0.5;
    col += 0.8*vec3(1.0,0.8,0.7)*pow(sun,512.0);
    col += 0.2*vec3(1.0,0.4,0.2)*pow(sun,32.0);
    col += 0.1*vec3(1.0,0.4,0.2)*pow(sun,4.0);
    
    vec3 bcol = col;
    
    // clouds
    float pt = (1000.0-ro.y)/rd.y; 
    if( pt>0.0 )
    {
        vec3 spos = ro + pt*rd;
        float clo = texture( _MainTex, 0.00006*spos.xz ).x;    
        vec3 cloCol = mix( vec3(0.4,0.5,0.6), vec3(1.3,0.6,0.4), pow(sun,2.0))*(0.5+0.5*clo);
        col = mix( col, cloCol, 0.5*smoothstep( 0.4, 1.0, clo ) );
    }
    
    // raymarch
    float tmax = 120.0;
    
    // bounding plane    
    float bt = (0.0-ro.y)/rd.y; 
    if( bt>0.0 ) tmax = min( tmax, bt );
        
    vec4 tmat = raycast( ro, rd, tmax);
    if( tmat.x<tmax )
    {
        // geometry
        vec3 pos = ro + tmat.x*rd;
        vec3 nor = calcNormal( pos, tmat.x );
        vec3 ref = reflect( rd, nor );

        float occ = smoothstep( 0.0, 1.5, pos.y + 11.5 ) * (1.0 - displacement( 0.25*pos*vec3(1.0,4.0,1.0) ));

        // materials
        vec4 mate = vec4(0.5,0.5,0.5,0.0);
        
        //if( tmat.z<0.5 )
        {
            vec3 uvw = 1.0*pos;

            vec3 bnor;
            float be = 1.0/1024.0;
            float bf = 0.4;
            bnor.x = texcube( _MainTex, bf*uvw+vec3(be,0.0,0.0), nor ).x - texcube( _MainTex, bf*uvw-vec3(be,0.0,0.0), nor ).x;
            bnor.y = texcube( _MainTex, bf*uvw+vec3(0.0,be,0.0), nor ).x - texcube( _MainTex, bf*uvw-vec3(0.0,be,0.0), nor ).x;
            bnor.z = texcube( _MainTex, bf*uvw+vec3(0.0,0.0,be), nor ).x - texcube( _MainTex, bf*uvw-vec3(0.0,0.0,be), nor ).x;
            bnor = normalize(bnor);
            float amo = 0.2  + 0.25*(1.0-smoothstep(0.6,0.7,nor.y) );
            nor = normalize( nor + amo*(bnor-nor*dot(bnor,nor)) );

            vec3 te = texcube( _MainTex, 0.15*uvw, nor ).xyz;
            te = 0.05 + te;
            mate.xyz = 0.6*te;
            mate.w = 1.5*(0.5+0.5*te.x);
            float th = smoothstep( 0.1, 0.4, texcube( _MainTex, 0.002*uvw, nor ).x );
            vec3 dcol = mix( vec3(0.2, 0.3, 0.0), 0.4*vec3(0.65, 0.4, 0.2), 0.2+0.8*th );
            mate.xyz = mix( mate.xyz, 2.0*dcol, th*smoothstep( 0.0, 1.0, nor.y ) );
            mate.xyz *= 0.5;
            float rr = smoothstep( 0.2, 0.4, texcube( _MainTex1, 2.0*0.02*uvw, nor ).y );
            mate.xyz *= mix( vec3(1.0,1,1), 1.5*vec3(0.25,0.24,0.22)*1.5, rr );
            mate.xyz *= 1.5*pow(texcube( _MainTex3, 8.0*uvw, nor ).xyz,vec3(0.5,.5,.5));
            mate = mix( mate, vec4(0.7,0.7,0.7,.0), smoothstep(0.8,0.9,nor.y + nor.x*0.6*te.x*te.x ));
            
            mate.xyz *= 1.5;
        }
        
        vec3 blig = normalize(vec3(-klig.x,0.0,-klig.z));
        vec3 slig = vec3( 0.0, 1.0, 0.0 );
            
        // lighting
        float sky = 0.0;
        sky += 0.2*diffuse( normalize(vec3( 0.0, 1.0, 0.0 )), nor, -rd, 1.0 );
        sky += 0.2*diffuse( normalize(vec3( 3.0, 1.0, 0.0 )), nor, -rd, 1.0 );
        sky += 0.2*diffuse( normalize(vec3(-3.0, 1.0, 0.0 )), nor, -rd, 1.0 );
        sky += 0.2*diffuse( normalize(vec3( 0.0, 1.0, 3.0 )), nor, -rd, 1.0 );
        sky += 0.2*diffuse( normalize(vec3( 0.0, 1.0,-3.0 )), nor, -rd, 1.0 );
        float dif = diffuse( klig, nor, -rd, 1.0 );
        float bac = diffuse( blig, nor, -rd, 1.0 );
        float sha = 0.0; if( dif>0.001 ) sha=softshadow( pos+0.01*nor, klig, 0.005, 64.0 );
        float spe = mate.w*pow( clamp(dot(reflect(rd,nor),klig),0.0,1.0),2.0)*clamp(dot(nor,klig),0.0,1.0);
        
        // lights
        vec3 lin = vec3(0.0,0,0);
        lin += 7.0*dif*vec3(1.20,0.50,0.25)*vec3(sha,sha*0.5+0.5*sha*sha, sha*sha );
        lin += 1.0*sky*vec3(0.10,0.50,0.70)*occ;
        lin += 2.0*bac*vec3(0.30,0.15,0.15)*occ;
        lin += 0.5*vec3(spe,spe,spe)*sha*occ;
        
        // surface-light interacion
        col = mate.xyz * lin;

        // fog
        bcol = 0.7*mix( vec3(0.2,0.5,1.0)*0.82, bcol, 0.15+0.8*sun ); col = mix( col, bcol, 1.0-exp(-0.02*tmat.x) );        
    }
    
    col += 0.15*vec3(1.0,0.9,0.6)*pow( sun, 6.0 );
    
    //-----------------------------------------------------
    // postprocessing
    //-----------------------------------------------------
//    col *= 1.0 - 0.25*pow(1.0-clamp(dot(cam[2],klig),0.0,1.0),3.0);
    
//    col = pow( max(col,0.0), vec3(0.45,0.45,0.45) );

//    col *= vec3(1.1,1.0,1.0);
//    col = clamp(col,0.0,1.0);
//    col = col*col*(3.0-2.0*col);
//    col = pow( col, vec3(0.9,1.0,1.0) );

 //   float dt = dot(col,vec3(0.333,.333,.333));
 //   col = mix( col, vec3(dt,dt,dt), 0.4 );
 //   col = col*0.5+0.5*mul(mul(col,col),(3.0-2.0*col));
    
 //   col *= 0.3 + 0.7*pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.1 );

 //   col *= smoothstep(0.0,2.5,iTime);

    fragColor = vec4( col, 1.0 );


                return fragColor;
            }

            ENDCG
        }
    }
}
