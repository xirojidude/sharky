
Shader "Skybox/Weather"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
        _MainTex2 ("tex2D", 2D) = "white" {}
        _XYZOffset ("XYZ", Color) = (0,0,0,0) 
        _XYZScale ("XYZ", Color) = (0,0,0,0) 
 
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
            uniform sampler2D _MainTex2; 
            float4 _XYZOffset;
            float4 _XYZScale;

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

// Weather. By David Hoskins, May 2014.
// @ https://www.shadertoy.com/view/4dsXWn
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// Who needs mathematically correct simulations?! :)
// It ray-casts to the bottom layer then steps through to the top layer.
// It uses the same number of steps for all positions.
// The larger steps at the horizon don't cause problems as they are far away.
// So the detail is where it matters.
// Unfortunately this can't be used to go through the cloud layer,
// but it's fast and has a massive draw distance.

float3 sunLight  = normalize( float3(  0.35, 0.14,  0.3 ) );
const float3 sunColour = float3(1.0, .7, .55);
float gTime, cloudy;
float3 flash;

#define CLOUD_LOWER 2800.0
#define CLOUD_UPPER 3800.0

#define tex2D_NOISE

#define MOD2 float2(.16632,.17369)
#define MOD3 float3(.16532,.17369,.15787)


//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
float Hash( float p )
{
    float2 p2 = frac(float2(p,p) * MOD2);
    p2 += dot(p2.yx, p2.xy+19.19);
    return frac(p2.x * p2.y);
}
float Hash(float3 p)
{
    p  = frac(p * MOD3);
    p += dot(p.xyz, p.yzx + 19.19);
    return frac(p.x * p.y * p.z);
}

//--------------------------------------------------------------------------
#ifdef tex2D_NOISE

//--------------------------------------------------------------------------
float Noise( in float2 f )
{
    float2 p = floor(f);
    f = frac(f);
    f = f*f*(3.0-2.0*f);
    float res = tex2D(_MainTex, (p+f+.5)/256.0).x; //tex2DLod(_MainTex, (p+f+.5)/256.0, 0.0).x;
    return res;
}
float Noise( in float3 x )
{
    #if 1
    return tex2D(_MainTex2, x*0.05).x;
    #else
    float3 p = floor(x);
    float3 f = frac(x);
    f = f*f*(3.0-2.0*f);
    
    float2 uv = (p.xy+float2(37.0,17.0)*p.z) + f.xy;
    float2 rg = tex2D( _MainTex, (uv+ 0.5)/256.0 ).yx;     //tex2DLod( _MainTex, (uv+ 0.5)/256.0, 0.0 ).yx;
    return lerp( rg.x, rg.y, f.z );
    #endif
}
#else

//--------------------------------------------------------------------------


float Noise( in float2 x )
{
    float2 p = floor(x);
    float2 f = frac(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    float res = lerp(lerp( Hash(n+  0.0), Hash(n+  1.0),f.x),
                    lerp( Hash(n+ 57.0), Hash(n+ 58.0),f.x),f.y);
    return res;
}
float Noise(in float3 p)
{
    float3 i = floor(p);
    float3 f = frac(p); 
    f *= f * (3.0-2.0*f);

    return lerp(
        lerp(lerp(Hash(i + float3(0.,0.,0.)), Hash(i + float3(1.,0.,0.)),f.x),
            lerp(Hash(i + float3(0.,1.,0.)), Hash(i + float3(1.,1.,0.)),f.x),
            f.y),
        lerp(lerp(Hash(i + float3(0.,0.,1.)), Hash(i + float3(1.,0.,1.)),f.x),
            lerp(Hash(i + float3(0.,1.,1.)), Hash(i + float3(1.,1.,1.)),f.x),
            f.y),
        f.z);
}
#endif

const float3x3 m = float3x3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
                    -0.60, -0.48,  0.64 ) * 1.7;
//--------------------------------------------------------------------------
float FBM( float3 p )
{
    p*= .0005;
    float f;
    
    f = 0.5000 * Noise(p); p = mul(m,p); //p.y -= gTime*.2;
    f += 0.2500 * Noise(p); p = mul(m,p); //p.y += gTime*.06;
    f += 0.1250 * Noise(p); p = mul(m,p);
    f += 0.0625   * Noise(p); p = mul(m,p);
    f += 0.03125  * Noise(p); p = mul(m,p);
    f += 0.015625 * Noise(p);
    return f;
}
//--------------------------------------------------------------------------
float FBMSH( float3 p )
{
    p*= .0005;
        
    float f;
    
    f = 0.5000 * Noise(p); p = mul(m,p); //p.y -= gTime*.2;
    f += 0.2500 * Noise(p); p = mul(m,p); //p.y += gTime*.06;
    f += 0.1250 * Noise(p); p = mul(m,p);
    f += 0.0625   * Noise(p); p = mul(m,p);
//  f += 0.03125  * Noise(p); p = mul(m,p);
/// f += 0.015625 * Noise(p);
    return f;
}

//--------------------------------------------------------------------------
float MapSH(float3 p)
{
    
    float h = -(FBM(p)-cloudy-.6);
    //h *= smoothstep(CLOUD_LOWER, CLOUD_LOWER+100., p.y);
    //h *= smoothstep(CLOUD_LOWER-500., CLOUD_LOWER, p.y);
    h *= smoothstep(CLOUD_UPPER+100., CLOUD_UPPER, p.y);
    return h;
}

//--------------------------------------------------------------------------

float SeaNoise( in float2 x )
{
    float2 p = floor(x);
    float2 f = frac(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    float res = lerp(lerp( Hash(n+  0.0), Hash(n+  1.0),f.x),
                    lerp( Hash(n+ 57.0), Hash(n+ 58.0),f.x),f.y);
    return res;
}
float SeaFBM( float2 p )
{
    p*= .001;
    float f;
    f = (sin(sin(p.x *1.22+gTime) + cos(p.y *.14)+p.x*.15+p.y*1.33-gTime)) * 1.0;
    
    f += (sin(p.x *.9+gTime + p.y *.3-gTime)) * 1.0;
    f += (cos(p.x *.7-gTime - p.y *.4-gTime)) * .5;
    f += 1.5000 * (.5-abs(SeaNoise(p)-.5)); p =  p * 2.05;
    f += .75000 * (.5-abs(SeaNoise(p)-.5)); p =  p * 2.02;
    f += 0.2500 * SeaNoise(p); p =  p * 2.07;
    f += 0.1250 * SeaNoise(p); p =  p * 2.13;
    //f += 0.0625 * Noise(p);

    return f;
}

//--------------------------------------------------------------------------
float Map(float3 p)
{
    float h = -(FBM(p)-cloudy-.6);
    
    return h;
}

//--------------------------------------------------------------------------
float SeaMap(in float2 pos)
{

    return SeaFBM(pos) * (20.0 + cloudy*170.0);
}

//--------------------------------------------------------------------------
float3 SeaNormal( in float3 pos, in float d, out float height)
{
    float p = .005 * d * d / 800;     // iResolution.x;
    float3 nor    = float3(0.0,         SeaMap(pos.xz), 0.0);
    float3 v2     = nor-float3(p,       SeaMap(pos.xz+float2(p,0.0)), 0.0);
    float3 v3     = nor-float3(0.0,     SeaMap(pos.xz+float2(0.0,-p)), -p);
    height = nor.y;
    nor = cross(v2, v3);
    return normalize(nor);
}

//--------------------------------------------------------------------------
float GetLighting(float3 p, float3 s)
{
    float l = MapSH(p)-MapSH(p+s*200.);
    return clamp(-l*2., 0.05, 1.0);
}



//--------------------------------------------------------------------------
// Grab all sky information for a given ray from camera
float3 GetSky(in float3 pos,in float3 rd, out float2 outPos)
{
    float sunAmount = max( dot( rd, sunLight), 0.0 );
    // Do the blue and sun...   
    float3  sky = lerp(float3(.0, .1, .4), float3(.3, .6, .8), 1.0-rd.y);
    sky = sky + sunColour * min(pow(sunAmount, 1500.0) * 5.0, 1.0);
    sky = sky + sunColour * min(pow(sunAmount, 10.0) * .6, 1.0);
    
    // Find the start and end of the cloud layer...
    float beg = ((CLOUD_LOWER-pos.y) / rd.y);
    float end = ((CLOUD_UPPER-pos.y) / rd.y);
    
    // Start position...
    float3 p = float3(pos.x + rd.x * beg, 0.0, pos.z + rd.z * beg);
    outPos = p.xz;
    beg +=  Hash(p)*150.0;

    // Trace clouds through that layer...
    float d = 0.0;
    float3 add = rd * ((end-beg) / 55.0);
    float2 shade;
    float2 shadeSum = float2(0.0, .0);
    shade.x = 1.0;
    // I think this is as small as the loop can be
    // for a reasonable cloud density illusion.
    [loop]
    for (int i = 0; i < 55; i++)
    {
        if (shadeSum.y >= 1.0) break;
        float h = Map(p);
        shade.y = max(h, 0.0); 

        shade.x = GetLighting(p, sunLight);

        shadeSum += shade * (1.0 - shadeSum.y);

        p += add;
    }
    //shadeSum.x /= 10.0;
    //shadeSum = min(shadeSum, 1.0);
    
    float3 clouds = lerp(float3(pow(shadeSum.x, .6),pow(shadeSum.x, .6),pow(shadeSum.x, .6)), sunColour, (1.0-shadeSum.y)*.4);
    //float3 clouds = float3(shadeSum.x);
    
    //clouds += min((1.0-sqrt(shadeSum.y)) * pow(sunAmount, 4.0), 1.0) * 2.0;
   
    clouds += flash * (shadeSum.y+shadeSum.x+.2) * .5;

    sky = lerp(sky, min(clouds, 1.0), shadeSum.y);
    
    return clamp(sky, 0.0, 1.0);
}

//--------------------------------------------------------------------------
float3 GetSea(in float3 pos,in float3 rd, out float2 outPos)
{
    float3 sea;
    float d = -pos.y/rd.y;
    float3 p = float3(pos.x + rd.x * d, 0.0, pos.z + rd.z * d);
    outPos = p.xz;
    
    float dis = length(p-pos);
    float h = 0.0;
    float3 nor = SeaNormal(p, dis, h);

    float3 ref = reflect(rd, nor);
    ref.y = max(ref.y, 0.0015);
    sea = GetSky(p, ref, p.xz);
    h = h*.005 / (1.0+max(dis*.02-300.0, 0.0));
    float fresnel = max(dot(nor, -rd),0.0);
    fresnel = pow(fresnel, .3)*1.1;
    
    sea = lerp(sea*.6, (float3(.3, .4, .45)+h*h) * max(dot(nor, sunLight), 0.0), min(fresnel, 1.0));
    
    float glit = max(dot(ref, sunLight), 0.0);
    sea += sunColour * pow(glit, 220.0) * max(-cloudy*100.0, 0.0);
    
    return sea;
}

float3 CameraPath( float t )
{
    return float3(4000.0 * sin(.16*t)+12290.0, 0.0, 8800.0 * cos(.145*t+.3));
} 

//--------------------------------------------------------------------------

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

    float m = 30; //(iMouse.x/iResolution.x)*30.0;
    gTime = _Time.y*.5 + m + 75.5;
    cloudy = cos(gTime * .25+.4) * .26;
    float lightning = 0.0;
    
    if (cloudy >= .2)
    {
        float f = ((gTime+1.5)% 2.5);
        if (f < .8)
        {
            f = smoothstep(.8, .0, f)* 1.5;
            lightning = ((-gTime*(1.5-Hash(gTime*.3)*.002))% 1.0) * f;
        }
    }
    
    flash = clamp(float3(1., 1.0, 1.2) * lightning, 0.0, 1.0);
       
    
    float2 xy = fragCoord.xy; // / iResolution.xy;
    float2 uv = (-1.0 + 2.0 * xy) ; //* float2(iResolution.x/iResolution.y,1.0);
    
    float3 cameraPos = ro+_XYZOffset.xyz; //CameraPath(gTime - 2.0);
//    float3 camTar    = ro ;CameraPath(gTime - .0);
//    camTar.y = cameraPos.y = sin(gTime) * 200.0 + 300.0;
//    camTar.y += 370.0;
    
//    float roll = .1 * sin(gTime * .25);
//    float3 cw = normalize(camTar-cameraPos);
//    float3 cp = float3(sin(roll), cos(roll),0.0);
//    float3 cu = cross(cw,cp);
//    float3 cv = cross(cu,cw);
    float3 dir; // = normalize(uv.x*cu + uv.y*cv + 1.3*cw);
//    float3x3 camMat = float3x3(cu, cv, cw);

    float3 col;
    float2 pos;

    cameraPos = ro+((_XYZOffset.xyz*100)-50);
    dir = rd * ((_XYZScale.xyz*10));
    pos = ro;//+((_XYZOffset.xyz*100)-50);
    if (dir.y > 0.0)
    {
        col = GetSky(cameraPos, dir, pos);
    }else
    {
        col = GetSea(cameraPos, dir, pos);
    }
    float l = exp(-length(pos) * .00002);
 //   col = lerp(float3(.6-cloudy*1.2,.6-cloudy*1.2,.6-cloudy*1.2)+flash*.3, col, max(l, .2));
    
    // Do the lens flares...
//    float bri = dot(cw, sunLight) * 2.7 * clamp(-cloudy+.2, 0.0, .2);
    if (false)   //(bri > 0.0)
    {
//        float2 sunPos = float2( dot( sunLight, cu ), dot( sunLight, cv ) );
//        float2 uvT = uv-sunPos;
//        uvT = uvT*(length(uvT));
//        bri = pow(bri, 6.0)*.6;

 //       float glare1 = max(1.2-length(uvT+sunPos*2.)*2.0, 0.0);
//        float glare2 = max(1.2-length(uvT+sunPos*.5)*4.0, 0.0);
//        uvT = lerp (uvT, uv, -2.3);
//        float glare3 = max(1.2-length(uvT+sunPos*5.0)*1.2, 0.0);

//        col += bri * sunColour * float3(1.0, .5, .2)  * pow(glare1, 10.0)*25.0;
//        col += bri * float3(.8, .8, 1.0) * pow(glare2, 8.0)*9.0;
//        col += bri * sunColour * pow(glare3, 4.0)*10.0;
    }
    
    float2 st =  uv * float2(.5+(xy.y+1.0)*.3, .02)+float2(gTime*.5+xy.y*.2, gTime*.2);
    // Rain...
#ifdef tex2D_NOISE
    float f = tex2D(_MainTex, st ).y * tex2D(_MainTex, st*.773).x * 1.55;
#else
    float f = Noise( st*200.5 ) * Noise( st*120.5 ) * 1.3;
#endif
    float rain = clamp(cloudy-.15, 0.0, 1.0);
    f = clamp(pow(abs(f), 15.0) * 5.0 * (rain*rain*125.0), 0.0, (xy.y+.1)*.6);
//    col = lerp(col, float3(0.15, .15, .15)+flash, f);
//    col = clamp(col, 0.0,1.0);

    // Stretch RGB upwards... 
    //col = (1.0 - exp(-col * 2.0)) * 1.1565;
    //col = (1.0 - exp(-col * 3.0)) * 1.052;
//    col = pow(col, float3(.7,.7,.7));
    //col = (col*col*(3.0-2.0*col));

    // Vignette...
//    col *= .55+0.45*pow(70.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.15 );   
    
    fragColor=float4(col, 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}


