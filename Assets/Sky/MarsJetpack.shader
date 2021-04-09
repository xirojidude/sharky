
Shader "Skybox/Empty"
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


// Mars Jetpack. By David Hoskins, December 2013.
// https://www.shadertoy.com/view/Md23Wz

// YouTube:-
// http://youtu.be/2eSb8zB4dBo

// Uses sphere tracing to accumulate direction normals across the landscape.
// Materials are calculated after the tracing loop,
// so only the normal can be used as reference.
// Sphere diameter to create depth of field is distance squared.

// For red/cyan 3D. Red on the left.
// #define STEREO

// Uncomment this for a faster landscape that uses a texture for the fracal:-
#define FASTER_LANDSCAPE

float3 sunLight  = normalize( float3(  0.35, 0.1,  0.3 ) );
const float3 sunColour = float3(1.0, .75, .5);
float2 coord;


//--------------------------------------------------------------------------
// Noise functions...
float Hash( float n )
{
    return frac(sin(n)*33753.545383);
}
float Linstep(float a, float b, float t)
{
    return clamp((t-a)/(b-a),0.,1.);

}

#ifdef FASTER_LANDSCAPE
//--------------------------------------------------------------------------

#define STEP (1.0/256.0)
float3 NoiseD( in float2 p )
{
    float2 f = frac(p);
    p = floor(p);
    float2 u = f*f*(1.5-f)*2.0;
    float4 n;
    n.x = tex2Dlod( _MainTex, float4((p+float2(0.5,0.5))*STEP,0, 0.0) ).x;
    n.y = tex2Dlod( _MainTex, float4((p+float2(1.5,0.5))*STEP,0, 0.0) ).x;
    n.z = tex2Dlod( _MainTex, float4((p+float2(0.5,1.5))*STEP,0, 0.0) ).x;
    n.w = tex2Dlod( _MainTex, float4((p+float2(1.5,1.5))*STEP,0, 0.0) ).x;

    // Normally you can make a texture out of these 4 so
    // you don't have to do any of it again...
    n.yzw = float3(n.x-n.y-n.z+n.w, n.y-n.x, n.z-n.x);
    float2 d = 6.0*f*(f-1.0)*(n.zw+n.y*u.yx);
    
    return float3(n.x + n.z * u.x + n.w * u.y + n.y * u.x * u.y, d.x, d.y);
}
#else



//--------------------------------------------------------------------------
float3 NoiseD( in float2 x )
{
    x+=4.2;
    float2 p = floor(x);
    float2 f = frac(x);

    float2 u = f*f*(3.0-2.0*f);
    //float2 u = f*f*f*(6.0*f*f - 15.0*f + 10.0);
    float n = p.x + p.y*57.0;

    float a = Hash(n+  0.0);
    float b = Hash(n+  1.0);
    float c = Hash(n+ 57.0);
    float d = Hash(n+ 58.0);
    return float3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
                6.0*f*(f-1.0)*(float2(b-a,c-a)+(a-b-c+d)*u.yx));
}
#endif
//--------------------------------------------------------------------------
#define START_HEIGHT 400.0
#define WARP  .15
#define SCALE  .002
#define HEIGHT 40.0
#define LACUNARITY 2.53
const float2x2 rotate2D = float2x2(1.732, 1.543, -1.543, 1.782);
float Terrain( in float2 p)
{
    p *= SCALE;
    float sum = 0.0;
    float freq = 1.;
    float amp = 3.5;
    float2 dsum = float2(0,0);
    for(int i=0; i < 5; i++)
    {
        float3 n = NoiseD(p + (WARP * dsum * freq));
        sum += amp * (1.0 - abs(n.x-.5)*2.0);
        dsum += amp * n.yz * -n.x;
        freq *= LACUNARITY;
        amp = amp*.5 * min(sum*.5, .9);
        p = mul(rotate2D , p);
    }
    return sum * HEIGHT;
    
}

//--------------------------------------------------------------------------
float Terrain2( in float2 p, in float sphereR)
{
    p *= SCALE;
    float sum = 0.0;
    float freq = 1.0;
    float amp = 3.5;
    float2 dsum = float2(0,0);
    for(int i=0; i < 8; i++)
    {
        float3 n = NoiseD(p + (WARP * dsum * freq));
        sum += amp * (1.0 - abs(n.x-.5)*2.0);
        dsum += amp * n.yz * -n.x;
        freq *= LACUNARITY;
        amp = amp * .5 * min(sum*.5, .9);
        p = mul(rotate2D , p);
    }
    return sum * HEIGHT;
}

//--------------------------------------------------------------------------
float Terrain3( in float2 p)
{
    p *= SCALE;
     float sum = 0.0;
     float freq = 1.0;
    float amp = 3.5;
     float2 dsum = float2(0,0);

     for(int i=0; i < 3; i++)
     {
        float3 n = NoiseD(p + (WARP * dsum * freq));
        sum += amp * (1.0 - abs(n.x-.5)*2.0);
        dsum += amp * n.yz * -n.x;
        freq *= LACUNARITY;
        amp = amp*.5 * min(sum*.5, .9);
        p = mul(rotate2D , p);
    }
    return sum * HEIGHT+20.0;

}


//--------------------------------------------------------------------------
float Map(in float3 p)
{
    float h = Terrain(p.xz);
    return p.y - h;
}

//--------------------------------------------------------------------------
// Grab all sky information for a given ray from camera
float3 GetSky(in float3 rd)
{
    float sunAmount = max( dot( rd, sunLight), 0.0 );
    float v = pow(1.0-max(rd.y,0.0),6.);
    float3  sky = lerp(float3(.015,0.0,.01), float3(.42, .2, .1), v);
    //sky *= smoothstep(-0.3, .0, rd.y);
    sky = sky + sunColour * sunAmount * sunAmount * .25;
    sky = sky + sunColour * min(pow(sunAmount, 800.0)*1.5, .3);
    return clamp(sky, 0.0, 1.0);
}

//--------------------------------------------------------------------------
float SphereRadius(float t)
{
    t = abs(t-250.0);
    t *= 0.01;
    return clamp(t*t, 50.0/450, 80.0); //iResolution.y, 80.0);
}

//--------------------------------------------------------------------------
// Calculate sun light...
float3 DoLighting(in float3 mat, in float3 normal, in float3 eyeDir)
{
    float h = dot(sunLight,normal);
    mat = mat * sunColour*(max(h, 0.0));
    mat += float3(0.04, .02,.02) * max(normal.y, 0.0);
    return mat;
}

//--------------------------------------------------------------------------
float3 GetNormal(float3 p, float sphereR)
{
    float2 j = float2(sphereR, 0.0);
    float3 nor    = float3(0.0,     Terrain2(p.xz, sphereR), 0.0);
    float3 v2     = nor-float3(j.x, Terrain2(p.xz+j, sphereR), 0.0);
    float3 v3     = nor-float3(0.0, Terrain2(p.xz-j.yx, sphereR), -j.x);
    nor = cross(v2, v3);
    return normalize(nor);
}

//--------------------------------------------------------------------------
float4 Scene(in float3 rO, in float3 rD)
{
    //float t = 0.0;
    float t = 20.0 * tex2D(_MainTex, coord.xy).y;
    float alpha;
    float4 normal = float4(0.0,0,0,0);
    float3 p = float3(0.0,0,0);
    float oldT = 0.0;
    for( int j=0; j < 105; j++ )
    {
        if (normal.w >= .8 || t > 1400.0) break;
        p = rO + t*rD;
        float sphereR = SphereRadius(t);
        float h = Map(p);
        if( h < sphereR)
        {
            // Accumulate the normals...
            //float3 nor = GetNormal(rO + BinarySubdivision(rO, rD, t, oldT, sphereR) * rD, sphereR);
            float3 nor = GetNormal(p, sphereR);
            alpha = (1.0 - normal.w) * ((sphereR-h) / sphereR);
            normal += float4(nor * alpha, alpha);
        }
        oldT = t;
        t +=  h*.5 + t * .003;
    }
    normal.xyz = normalize(normal.xyz);
    // Scale the alpha up to 1.0...
    normal.w = clamp(normal.w * (1.0 / .8), 0.0, 1.0);
    // Fog...   :)
    normal.w /= 1.0+(smoothstep(300.0, 1400.0, t) * 2.0);
    return normal;
}

//--------------------------------------------------------------------------
float3 CameraPath( float t )
{
    float2 p = float2(400.0 * sin(3.54*t), 400.0 * cos(2.0*t) );
    return float3(p.x+440.0,  0.0, p.y+10.0);
} 

float Hash(float2 p)
{
    return frac(sin(dot(p, float2(12.9898, 78.233))) * 33758.5453)-.5;
}

//--------------------------------------------------------------------------
float3 PostEffects(float3 rgb, float2 xy)
{
    // Gamma first...
    rgb = pow(rgb, float3(0.45,0.45,0.45));

    // Then...
    #define CONTRAST 1.2
    #define SATURATION 1.3
    #define BRIGHTNESS 1.4
    float dt = dot(float3(.2125, .7154, .0721), rgb*BRIGHTNESS);
    rgb = lerp(float3(.5,.5,.5), lerp(float3(dt,dt,dt), rgb*BRIGHTNESS, SATURATION), CONTRAST);
    // Noise...
    // rgb = clamp(rgb+Hash(xy*_Time.y)*.1, 0.0, 1.0);
    // Vignette...
    rgb *= .4+0.5*pow(40.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2 );  

    return rgb;
}

float mod(float a, float b) {return a%b;}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

//   float m = (iMouse.x/iResolution.x)*300.0;
    float gTime = (_Time.y*8.0+2321.0)*.006;
//    float2 xy = fragCoord.xy / iResolution.xy;
//    float2 uv = (-1.0 + 2.0 * xy) * float2(iResolution.x/iResolution.y,1.0);
    coord = fragCoord;  // / iChannelResolution[0].xy;
    float3 camTar;
    
    float hTime = mod(gTime+1.95, 2.0);
    
    #ifdef STEREO
//    float isRed = mod(fragCoord.x + mod(fragCoord.y,2.0),2.0);
    #endif

    float3 cameraPos = ro; //CameraPath(gTime + 0.0);

    //float height = 300.-hTime*24.0;
//    float height = (smoothstep(.3, 0.0, hTime) + smoothstep(1.7, 2.0, hTime)) * 400.0;
//    camTar   = CameraPath(gTime + .3);
//    cameraPos.y += height;
    
    float t = Terrain3(CameraPath(gTime + .009).xz)+20.0;
//    if (cameraPos.y < t) cameraPos.y = t;
//    camTar.y = cameraPos.y-clamp(height-40.0, 0.0, 100.0);

//    float roll = .4*sin(gTime+.5);
//    float3 cw = normalize(camTar-cameraPos);
//    float3 cp = float3(sin(roll), cos(roll),0.0);
//    float3 cu = cross(cw,cp);
//    float3 cv = cross(cu,cw);
    float3 dir = rd;  //normalize(uv.x*cu + uv.y*cv + 1.1*cw);
//    float3x3 camMat = float3x3(cu, cv, cw);

    #ifdef STEREO
//    cameraPos += 1.5*cu*isRed; // move camera to the right - the rd Vector is still good
    #endif

    float3 col;
    float distance;
    float4 normal;
    normal = Scene(cameraPos, dir);
    
    col = lerp(float3(.4, 0.5, 0.5), float3(.7, .35, .1),smoothstep(0.8, 1.1, (normal.y)));
    col = lerp(col, float3(0.17, 0.05, 0.0), clamp(normal.z+.2, 0.0, 1.0));
    col = lerp(col, float3(.8, .8,.5), clamp((normal.x-.6)*1.3, 0.0, 1.0));

    if (normal.w > 0.0) col = DoLighting(col, normal.xyz, dir);

    col = lerp(GetSky(dir), col, normal.w);

/*
    // bri is the brightness of sun at the centre of the camera direction.
    // Yeah, the lens flares is not exactly subtle, but it was good fun making it.
    float bri = dot(cw, sunLight)*.7;
    if (bri > 0.0)
    {
        float2 sunPos = float2( dot( sunLight, cu ), dot( sunLight, cv ) );
        float2 uvT = uv-sunPos;
        uvT = uvT*(length(uvT));
        bri = pow(bri, 6.0)*.8;

        // glare = the red shifted blob...
        float glare1 = max(dot(normalize(float3(dir.x, dir.y+.3, dir.z)),sunLight),0.0)*1.4;
        // glare2 is the yellow ring...
        float glare2 = max(1.0-length(uvT+sunPos*.5)*4.0, 0.0);
        uvT = lerp (uvT, uv, -2.3);
        // glare3 is a purple splodge...
        float glare3 = max(1.0-length(uvT+sunPos*5.0)*1.2, 0.0);

        col += bri * float3(1.0, .0, .0)  * pow(glare1, 12.5)*.05;
        col += bri * float3(1.0, .5, 0.5) * pow(glare2, 2.0)*2.5;
        col += bri * sunColour * pow(glare3, 2.0)*3.0;
    }
*/
    col = PostEffects(col, float2(1,1)); 
    
    #ifdef STEREO   
//    col *= float3( isRed, 1.0-isRed, 1.0-isRed ); 
    #endif
    
    fragColor=float4(col,1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}
