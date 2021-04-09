
Shader "Skybox/GrassyHillis"
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


// Rolling hills. By David Hoskins, November 2013.
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// https://www.shadertoy.com/view/Xsf3zX

// v.2.00 Uses eiffie's 'Circle of Confusion' function
//        for blurred ray marching into the grass.
// v.1.02 Camera aberrations.
// v.1.01 Added better grass, with wind movement.

// For red/cyan 3D...
//#define STEREO

#define MOD2 float2(3.07965, 7.4235)
float PI  = 4.0*atan(1.0);
float3 sunLight  = normalize( float3(  0.35, 0.2,  0.3 ) );
float3 cameraPos;
float3 sunColour = float3(1.0, .75, .6);
const float2x2 rotate2D = float2x2(1.932, 1.623, -1.623, 1.952);
float gTime = 0.0;

//--------------------------------------------------------------------------
// Noise functions...
float Hash( float p )
{
    float2 p2 = frac(float2(p,p) / MOD2);
    p2 += dot(p2.yx, p2.xy+19.19);
    return frac(p2.x * p2.y);
}

//--------------------------------------------------------------------------
float Hash(float2 p)
{
    p  = frac(p / MOD2);
    p += dot(p.xy, p.yx+19.19);
    return frac(p.x * p.y);
}


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

float2 Voronoi( in float2 x )
{
    float2 p = floor( x );
    float2 f = frac( x );
    float res=100.0,id;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        float2 b = float2( float(i), float(j) );
        float2 r = float2( b ) - f  + Hash( p + b );
        float d = dot(r,r);
        if( d < res )
        {
            res = d;
            id  = Hash(p+b);
        }           
    }
    return float2(max(.4-sqrt(res), 0.0),id);
}


//--------------------------------------------------------------------------
float2 Terrain( in float2 p)
{
    float type = 0.0;
    float2 pos = p*0.003;
    float w = 50.0;
    float f = .0;
    for (int i = 0; i < 3; i++)
    {
        f += Noise(pos) * w;
        w = w * 0.62;
        pos *= 2.5;
    }

    return float2(f, type);
}

//--------------------------------------------------------------------------
float2 Map(in float3 p)
{
    float2 h = Terrain(p.xz);
    return float2(p.y - h.x, h.y);
}

//--------------------------------------------------------------------------
float fracalNoise(in float2 xy)
{
    float w = .7;
    float f = 0.0;

    for (int i = 0; i < 3; i++)
    {
        f += Noise(xy) * w;
        w = w*0.6;
        xy = 2.0 * xy;
    }
    return f;
}

//--------------------------------------------------------------------------
// Grab all sky information for a given ray from camera
float3 GetSky(in float3 rd)
{
    float sunAmount = max( dot( rd, normalize(_SunDir.xyz)),0   ); //sunLight), 0.0 );
    float v = pow(1.0-max(rd.y,0.0),6.);
    float3  sky = lerp(float3(.1, .2, .3), float3(.32, .32, .32), v);
    sky = sky + sunColour * sunAmount * sunAmount * .25;
    sky = sky + sunColour * min(pow(sunAmount, 800.0)*1.5, .3);
    return clamp(sky, 0.0, 1.0);
}

//--------------------------------------------------------------------------
// Merge grass into the sky background for correct fog colouring...
float3 ApplyFog( in float3  rgb, in float dis, in float3 dir)
{
    float fogAmount = clamp(dis*dis* 0.0000012, 0.0, 1.0);
    return lerp( rgb, GetSky(dir), fogAmount );
}

//--------------------------------------------------------------------------
float3 DE(float3 p)
{
    float base = Terrain(p.xz).x - 1.9;
    float height = Noise(p.xz*2.0)*.75 + Noise(p.xz)*.35 + Noise(p.xz*.5)*.2;
    //p.y += height;
    float y = p.y - base-height;
    y = y*y;
    float2 ret = Voronoi((p.xz*2.5+sin(y*4.0+p.zx*12.3)*.12+float2(sin(_Time.y*2.3+1.5*p.z),sin(_Time.y*3.6+1.5*p.x))*y*.5));
    float f = ret.x * .6 + y * .58;
    return float3( y - f*1.4, clamp(f * 1.5, 0.0, 1.0), ret.y);
}

//--------------------------------------------------------------------------
// eiffie's code for calculating the aperture size for a given distance...
float CircleOfConfusion(float t)
{
    return max(t * .04, (2.0 / 450) * (1.0+t));   //iResolution.y) * (1.0+t));
}

//--------------------------------------------------------------------------
float Linstep(float a, float b, float t)
{
    return clamp((t-a)/(b-a),0.,1.);
}

//--------------------------------------------------------------------------
float3 GrassBlades(in float3 rO, in float3 rD, in float3 mat, in float dist)
{
    float d = 0.0;
    // Only calculate cCoC once is enough here...
    float rCoC = CircleOfConfusion(dist*.3);
    float alpha = 0.0;
    
    float4 col = float4(mat*0.15, 0.0);

    for (int i = 0; i < 15; i++)
    {
        if (col.w > .99) break;
        float3 p = rO + rD * d;
        
        float3 ret = DE(p);
        ret.x += .5 * rCoC;

        if (ret.x < rCoC)
        {
            alpha = (1.0 - col.y) * Linstep(-rCoC, rCoC, -ret.x);//calculate the lerp like cloud density
            // lerp material with white tips for grass...
            float3 gra = lerp(mat, float3(.35, .35, min(pow(ret.z, 4.0)*35.0, .35)), pow(ret.y, 9.0)*.7) * ret.y;
            col += float4(gra * alpha, alpha);
        }
        d += max(ret.x * .7, .1);
    }
    if(col.w < .2)
        col.xyz = float3(0.1, .15, 0.05);
    return col.xyz;
}

//--------------------------------------------------------------------------
// Calculate sun light...
void DoLighting(inout float3 mat, in float3 pos, in float3 normal, in float3 eyeDir, in float dis)
{
    float h = dot(_SunDir,normal);     //dot(sunLight,normal);
    mat = mat * float3(1.0, .75, .6) *(max(h, 0.0)+.2); //sunColour;  //
}

//--------------------------------------------------------------------------
float3 TerrainColour(float3 pos, float3 dir,  float3 normal, float dis, float type)
{
    float3 mat;
    if (type == 0.0)
    {
        // Random colour...
        mat = lerp(float3(.0,.3,.0), float3(.2,.3,.0), Noise(pos.xz*.025));
        // Random shadows...
        float t = fracalNoise(pos.xz * .1)+.5;
        // Do grass blade tracing...
        mat = GrassBlades(pos, dir, mat, dis) * t;
        DoLighting(mat, pos, normal,dir, dis);
    }
    mat = ApplyFog(mat, dis, dir);
    return mat;
}

//--------------------------------------------------------------------------
// Home in on the surface by dividing by two and split...
float BinarySubdivision(in float3 rO, in float3 rD, float t, float oldT)
{
    float halfwayT = 0.0;
    for (int n = 0; n < 5; n++)
    {
        halfwayT = (oldT + t ) * .5;
        if (Map(rO + halfwayT*rD).x < .05)
        {
            t = halfwayT;
        }else
        {
            oldT = halfwayT;
        }
    }
    return t;
}

//--------------------------------------------------------------------------
bool Scene(in float3 rO, in float3 rD, out float resT, out float type )
{
    float t = 5.;
    float oldT = 0.0;
    float delta = 0.;
    float2 h = float2(1.0, 1.0);
    bool hit = false;
    for( int j=0; j < 70; j++ )
    {
        float3 p = rO + t*rD;
        h = Map(p); // ...Get this position's height mapping.

        // Are we inside, and close enough to fudge a hit?...
        if( h.x < 0.05)
        {
            hit = true;
            break;
        }
            
        delta = h.x + (t*0.03);
        oldT = t;
        t += delta;
    }
    type = h.y;
    resT = BinarySubdivision(rO, rD, t, oldT);
    return hit;
}

//--------------------------------------------------------------------------
float3 CameraPath( float t )
{
    //t = time + t;
    float2 p = float2(200.0 * sin(3.54*t), 200.0 * cos(2.0*t) );
    return float3(p.x+55.0,  12.0+sin(t*.3)*6.5, -94.0+p.y);
} 

//--------------------------------------------------------------------------
float3 PostEffect(float3 rgb, float2 xy)
{
    // Gamma first...
    rgb = pow(rgb, float3(0.45,0.45,0.45));
    
    // Then...
    #define CONTRAST 1.1
    #define SATURATION 1.3
    #define BRIGHTNESS 1.3
    float dt = dot(float3(.2125, .7154, .0721), rgb*BRIGHTNESS);
    rgb = lerp(float3(.5,.5,.5), lerp(float3(dt,dt,dt), rgb*BRIGHTNESS, SATURATION), CONTRAST);
    // Vignette...
//    rgb *= .4+0.5*pow(40.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2 );  
    return rgb;
}


float mod (float a, float b) {return a%b;}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

//   float m = (iMouse.x/iResolution.x)*300.0;
    float gTime = (_Time.y*5.0+2352.0)*.006;
//    float2 xy = fragCoord.xy / iResolution.xy;
    float2 uv = fragCoord.xy; //(-1.0 + 2.0 * xy) * float2(iResolution.x/iResolution.y,1.0);
    float3 camTar;
    
//    if (xy.y < .13 || xy.y >= .87)
//    {
//        // Top and bottom cine-crop - what a waste! :)
//        fragColor=float4(float4(0.0));
//        return;
//    }


    #ifdef STEREO
//    float isCyan = mod(fragCoord.x + mod(fragCoord.y,2.0),2.0);
    #endif

    cameraPos = ro; //CameraPath(gTime + 0.0);
//    cameraPos.x -= 3.0;
    camTar   = ro + (rd*.1); //CameraPath(gTime + .009);
//    cameraPos.y += Terrain(CameraPath(gTime + .009).xz).x;
//    camTar.y = cameraPos.y;
    
    float roll = .4*sin(gTime+.5);
    float3 cw = normalize(camTar-cameraPos);
    float3 cp = float3(sin(roll), cos(roll),0.0);
    float3 cu = cross(cw,cp);
    float3 cv = cross(cu,cw);
    float3 dir = rd;  //normalize(uv.x*cu + uv.y*cv + 1.3*cw);
    float3x3 camMat = float3x3(cu, cv, cw);

    #ifdef STEREO
//    cameraPos += .85*cu*isCyan; // move camera to the right - the rd Vector is still good
    #endif

    float3 col;
    float distance;
    float type;
    if( !Scene(cameraPos, dir, distance, type) )
    {
        // Missed scene, now just get the sky...
        col = GetSky(dir);
    }
    else
    {
        // Get world coordinate of landscape...
        float3 pos = cameraPos + distance * dir;
        // Get normal from sampling the high definition height map
        // Use the distance to sample larger gaps to help stop aliasing...
        float2 p = float2(0.1, 0.0);
        float3 nor    = float3(0.0,     Terrain(pos.xz).x, 0.0);
        float3 v2     = nor-float3(p.x, Terrain(pos.xz+p).x, 0.0);
        float3 v3     = nor-float3(0.0, Terrain(pos.xz-p.yx).x, -p.x);
        nor = cross(v2, v3);
        nor = normalize(nor);

        // Get the colour using all available data...
        col = TerrainColour(pos, dir, nor, distance, type);
    }
    
    // bri is the brightness of sun at the centre of the camera direction.
    // Yeah, the lens flares is not exactly subtle, but it was good fun making it.
    float bri = dot(cw, sunLight)*.75;
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
        col += bri * float3(1.0, 1.0, 0.2) * pow(glare2, 2.0)*2.5;
        col += bri * sunColour * pow(glare3, 2.0)*3.0;
    }
    col = PostEffect(col, float2(0,0)); //xy); 
    
    #ifdef STEREO   
//    col *= float3( isCyan, 1.0-isCyan, 1.0-isCyan );  
    #endif
    
    fragColor=float4(col,1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}


