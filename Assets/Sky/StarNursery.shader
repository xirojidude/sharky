
Shader "Skybox/StarNursery"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
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

            uniform sampler2D _MainTex; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }


// Built from the basics of'Clouds' Created by inigo quilez - iq/2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// Edited by Dave Hoskins into "Star Nursery"
// V.1.1 Some speed up in the ray-marching loop.
// V.1.2 Added Shadertoy's fast 3D noise for better, smaller step size.

float3x3 m = float3x3( 0.30,  0.90,  0.60,
              -0.90,  0.36, -0.48,
              -0.60, -0.48,  0.34 );
#define time (_Time.y+46.0)

//----------------------------------------------------------------------
float hash( float n )
{
    return frac(sin(n)*43758.5453123);
}

//----------------------------------------------------------------------
float noise( in float2 x )
{
    float2 p = floor(x);
    float2 f = frac(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0;

    float res = lerp(lerp( hash(n+  0.0), hash(n+  1.0),f.x),
                    lerp( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);

    return res;
}

//----------------------------------------------------------------------
float noise( in float3 x )
{
    #if 0

    // 3D tex2D
    return tex2D(_MainTex2, x*.03).x*1.05;
    
    #else
    
    // Use 2D tex2D...
    float3 p = floor(x);
    float3 f = frac(x);
    f = f*f*(3.0-2.0*f);

    float2 uv = (p.xy+float2(37.0,17.0)*p.z) + f.xy;
    float2 rg = tex2D( _MainTex, (uv+ 0.5)/256.0).yx;  //, -99.0).yx;
    return lerp( rg.x, rg.y, f.z );
    
    #endif
}

//----------------------------------------------------------------------
float fbm( float3 p )
{
    float f;
    f  = 1.600*noise( p ); p = mul(m,p*2.02);
    f += 0.3500*noise( p ); p = mul(m,p*2.33);
    f += 0.2250*noise( p ); p = mul(m,p*2.03);
    f += 0.0825*noise( p ); p = mul(m,p*2.01);
    return f;
}

//----------------------------------------------------------------------
float4 map( in float3 p )
{
    float d = 0.01- p.y;

    float f= fbm( p*1.0 - float3(.4,0.3,-0.3)*time);
    d += 4.0 * f;

    d = clamp( d, 0.0, 1.0 );
    
    float4 res = float4( d,d,d,d );
    res.w = pow(res.y, .1);

    res.xyz = lerp( .7*float3(1.0,0.4,0.2), float3(0.2,0.0,0.2), res.y * 1.);
    res.xyz = res.xyz + pow(abs(.95-f), 26.0) * 1.85;
    return res;
}


//----------------------------------------------------------------------
float3 sundir = float3(1.0,0.4,0.0);
float4 raymarch( in float3 ro, in float3 rd )
{
    float4 sum = float4(0, 0, 0, 0);

    float t = 0.0;
    float3 pos = float3(0.0, 0.0, 0.0);
    for(int i=0; i<100; i++)
    {
        if (sum.a > 0.8 || pos.y > 9.0 || pos.y < -2.0) continue;
        pos = ro + t*rd;

        float4 col = map( pos );
        
        // Accumulate the alpha with the colour...
        col.a *= 0.08;
        col.rgb *= col.a;

        sum = sum + col*(1.0 - sum.a);  
        t += max(0.1,0.04*t);
    }
    sum.xyz /= (0.003+sum.w);

    return clamp( sum, 0.0, 1.0 );
}


         fixed4 frag (v2f v2) : SV_Target
            {
                float2 fragCoord = v2.vertex;

                float3 viewDirection = normalize(v2.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v2.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

//    float2 q = fragCoord.xy / iResolution.xy;
//    float2 p = -1.0 + 2.0*q;
 //   p.x *= iResolution.x/ iResolution.y;
//    float2 mo = (-1.0 + 2.0 + iMouse.xy) / iResolution.xy;
    
    // Camera code...
     ro = ro+ 5.6*normalize(float3(cos(2.75-3.0), .4-1.3*(-2.4), sin(2.75-2.0)));
//    float3 ta = float3(.0, 5.6, 2.4);
//    float3 ww = normalize( ta - ro);
//    float3 uu = normalize(cross( float3(0.0,1.0,0.0), ww ));
//    float3 vv = normalize(cross(ww,uu));
//    float3 rd = normalize( p.x*uu + p.y*vv + 1.5*ww );

    // Ray march into the clouds adding up colour...
    float4 res = raymarch( ro, rd );
    

    float sun = clamp( dot(sundir,rd), 0.0, 2.0 );
    float3 col = lerp(float3(.3,0.0,0.05), float3(0.2,0.2,0.3), sqrt(max(rd.y, 0.001)));
    col += .4*float3(.4,.2,0.67)*sun;
    col = clamp(col, 0.0, 1.0);
    col += 0.43*float3(.4,0.4,0.2)*pow( sun, 21.0 );
    
    // Do the stars...
    float v = 1.0/( 2. * ( 1. + rd.z ) );
    float2 xy = float2(rd.y * v, rd.x * v);
    rd.z += time*.002;
    float s = noise(rd.xz*134.0);
    s += noise(rd.xz*370.);
    s += noise(rd.xz*870.);
    s = pow(s,19.0) * 0.00000001 * max(rd.y, 0.0);
    if (s > 0.0)
    {
        float3 backStars = float3((1.0-sin(xy.x*20.0+time*13.0*rd.x+xy.y*30.0))*.5*s,s, s); 
        col += backStars;
    }

    // lerp in the clouds...
    col = lerp( col, res.xyz, res.w*1.3);
    
    #define CONTRAST 1.1
    #define SATURATION 1.15
    #define BRIGHTNESS 1.03
    float newcol = dot(float3(.2125, .7154, .0721), col*BRIGHTNESS);
    col = lerp(float3(.5,.5,.5), lerp(float3(newcol,newcol,newcol), col*BRIGHTNESS, SATURATION), CONTRAST);
    
    fragColor = float4( col, 1.0 );

                return fragColor;
            }

            ENDCG
        }
    }
}






