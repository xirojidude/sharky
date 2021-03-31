

Shader "Skybox/Alps"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "LightMode"="Always" }
        LOD 1000

        Pass
        {
            Lighting Off
            AlphaToMask Off
            Cull Off

            //Cull Back
            
            //ZTest Always
            //Blend One Zero

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #define UNITY_PASS_FORWARDBASE
            #pragma multi_compile_fwdbase_fullshadows
            #pragma only_renderers d3d9 d3d11 glcore gles 
            #pragma target 3.0
            #pragma shader_feature FANCY_STUFF_OFF
            #pragma glsl_es2

            
 

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


// Alps.
// by David Hoskins.
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// https://www.shadertoy.com/view/4ssXW2
// Uses a ridged fracal noise with corrosion effects for higher altitudes.

//#define STEREO   // RED left eye.
//#define SHADOWS  // fake shadows.
#define MOD3 float3(.0631,.07369,.08787)
//#define MOD4 float4(.0631,.07369,.08787, .09987)
float3 sunLight  = normalize( float3(  -0.2, 0.2,  -1.0 ) );
float3 sunColour = float3(1.0, .88, .75);
float specular = 0.0;
float3 cameraPos;
float ambient;
const float2 add = float2(1.0,0.0);

// This peturbs the fracal positions for each iteration down...
// Helps make nice twisted landscapes...
const float2x2 rotate2D = float2x2(1.6623, 1.850, -1.7131, 1.4623);

//--------------------------------------------------------------------------
// Noise functions...
//----------------------------------------------------------------------------------------
float Hash12(float2 p)
{
    float3 p3  = frac(float3(p.xyx) * MOD3);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac(p3.x * p3.y * p3.z);
}
//--------------------------------------------------------------------------
float Noise( in float2 x )
{
    float2 p = floor(x);
    float2 f = frac(x);
    f = f*f*(1.5-f)*2.0;
    
    float res = lerp(lerp( Hash12(p), Hash12(p + add.xy),f.x),
                    lerp( Hash12(p + add.yx), Hash12(p + add.xx),f.x),f.y);
    return res;
}

//--------------------------------------------------------------------------
float3 NoiseD( in float2 x )
{
    x+=4.2;
    float2 p = floor(x);
    float2 f = frac(x);

    float2 u = f*f*(1.5-f)*2.0;;
    
    float a = Hash12(p);
    float b = Hash12(p + add.xy);
    float c = Hash12(p + add.yx);
    float d = Hash12(p + add.xx);
    return float3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
                6.0*f*(f-1.0)*(float2(b-a,c-a)+(a-b-c+d)*u.yx));
}

float Noise( in float3 x )
{
    float3 p = floor(x);
    float3 f = frac(x);
    f = f*f*(1.5-f)*2.0;
    
    float2 uv = (p.xy+float2(37.0,17.0)*p.z) + f.xy;
    float2 rg = tex2D( _MainTex, (uv+ 0.5)/256.0 ).yx;
    return lerp( rg.x, rg.y, f.z );
}
#define WARP  .15
#define SCALE  .0023
#define HEIGHT 55.
#define LACUNARITY 2.13
//--------------------------------------------------------------------------
// Low-def version for ray-marching through the height field...
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
        amp = amp*.4 * min(sum*.22, 1.0);
        p = mul(rotate2D , p);
    }
    return sum * HEIGHT;    
}

//--------------------------------------------------------------------------
// High-def version only used for grabbing normal information....
float Terrain2( in float2 p, in float d)
{
    int stop = 1+int(9.0-d*.000004);
    p *= SCALE;
    float sum = 0.0;
    float freq = 1.;
    float amp = 3.5;
    float2 dsum = float2(0,0);
    float3 n;
    for(int i=0; i < 9; i++)
    {
        if (i > stop) break;
        n = NoiseD(p + (WARP * dsum * freq));
        sum += amp * (1.0 - abs(n.x-.5)*2.0);
        dsum += amp * n.yz * -n.x;
        freq *= LACUNARITY;
        amp = amp*.4 * min(sum*.22, 1.0);
        sum += n.x/(222.5+dot(dsum, dsum));
        p = mul(rotate2D , p);
    }
    return sum * HEIGHT;
}

// Low detailed camera version... 
float TerrainCam( in float2 p)
{
    
    p *= SCALE;
    float sum = 0.0;
    float freq = 1.;
    float amp = 3.5;
    float2 dsum = float2(0,0);
    for(int i=0; i < 2; i++)
    {
        float3 n = NoiseD(p + (WARP * dsum * freq));
        sum += amp * (1.0 - abs(n.x-.5)*2.0);
        dsum += amp * n.yz * -n.x;
        freq *= LACUNARITY;
        amp = amp*.4 * min(sum*.22, 1.0);
        p = mul(rotate2D , p);
    }
    return sum * HEIGHT;
    
}

float FBM( float3 p )
{
    
    p *= .015;
    p.xz *= .3;
    //p.zy -= _Time.y * .04;
    
    float f;
    f  = 0.5000  * Noise(p); p = p * 3.02; //p.y -= gTime*.2;
    f += 0.2500  * Noise(p); p = p * 3.03; //p.y += gTime*.06;
    f += 0.1250  * Noise(p); p = p * 4.01;
    f += 0.0625  * Noise(p); p = p * 4.023;
    //f += 0.03125 * Noise(p);
    return f;
}


//--------------------------------------------------------------------------
// Map to lower resolution for height field mapping for Scene function...
float Map(in float3 p)
{
    float h = Terrain(p.xz);
    return p.y - h;
}

//--------------------------------------------------------------------------
float MapClouds(float3 p)
{
    float h = FBM(p)*1.0;
    return (-h+.6);// + (p.y)*.0002);
}
//--------------------------------------------------------------------------
// Grab all sky information for a given ray from camera
float3 GetSky(in float3 rd)
{
    float v = pow(1.0-max(rd.y,0.0),10.);
    float3  sky = float3(v*sunColour.x*0.42+.04, v*sunColour.y*0.4+0.09, v*sunColour.z*0.4+.17);
    return sky;
}

//--------------------------------------------------------------------------
// Merge mountains into the sky background for correct disappearance...
float3 ApplyFog( in float3  rgb, in float dis, in float3 dir)
{
    return lerp(GetSky(dir), rgb, exp(-.000001*dis) );
}

//--------------------------------------------------------------------------
// Calculate sun light...
void DoLighting(inout float3 mat, in float3 pos, in float3 normal, in float3 eyeDir, in float dis)
{
    float h = dot(sunLight,normal);

#ifdef SHADOWS
    float3  eps = float3(1.0,0.0,0.0);
    float3 nor;
    nor.x = Terrain(pos.xz-eps.xy) - Terrain(pos.xz+eps.xy);
    nor.y = 1.0*eps.x;
    nor.z = Terrain(pos.xz-eps.yx) - Terrain(pos.xz+eps.yx);
    nor = normalize(nor);
    float shad = clamp(1.0*dot(nor,sunLight), 0.0, 1.0 );
    float c = max(h, 0.0) * shad;
#else
    float c = max(h, 0.0);
#endif    
    float3 R = reflect(sunLight, normal);
    mat = mat * sunColour * c * float3(.9, .9, 1.0) +  GetSky(R)*ambient;
    // Specular...
    if (h > 0.0)
    {
        float specAmount = pow( max(dot(R, normalize(eyeDir)), 0.0), 34.0)*specular;
        mat += sunColour * specAmount;
    }
}

//--------------------------------------------------------------------------
float3 TerrainColour(float3 pos, float3 normal, float dis)
{
    float3 mat;
    specular = .0;
    ambient = .4 * abs(normal.y);
    float3 dir = normalize(pos-cameraPos);
    
    float disSqrd = dis * dis;// Squaring it gives better distance scales.

    float f = clamp(Noise(pos.xz*.001), 0.0,1.0);//*10.8;
    f *= Noise(pos.zx*.2+normal.xz*1.5);
    //f *= .5;
    mat = lerp(float3(.1,.1,.1), float3(.1, .07, .01), f);

    // Snow...
    if (pos.y > 75.0 && normal.y > .2)
    {
        float snow = smoothstep(0.0, 1.0, (pos.y - 75.0 - Noise(pos.xz * .3)*Noise(pos.xz * .027)*83.0) * 0.2 * (normal.y));
        mat = lerp(mat, float3(.8,.9,1.0), min(snow, 1.0));
        specular += snow*.7;
        ambient+=snow *.05;
    }

    DoLighting(mat, pos, normal,dir, disSqrd);
    
    mat = ApplyFog(mat, disSqrd, dir);
    return mat;
}

//--------------------------------------------------------------------------
float BinarySubdivision(in float3 rO, in float3 rD, float2 dist)
{
    // Home in on the surface by dividing by two and split...
    float2 newdist = dist;
    for (int n = 0; n < 6; n++)
    {
        float halfwayT = (newdist.x + newdist.y) * .5;
        float3 p = rO + halfwayT * rD;
        if (Map(p) < .5)  newdist.x = halfwayT;
         else newdist.y = halfwayT;
    }
    return newdist.x;
}

//--------------------------------------------------------------------------
bool Scene(in float3 rO, in float3 rD, in float2 uv, out float resT, out float2 cloud)
{
    float t = 10.0 + Hash12(uv*3333.0)* 13.0;
    float oldT = 0.0;
    float delta = 0.0;
    bool fin = false;
    float2 distances;
    float2 shade = cloud = float2(0.0, 0.0);
    float3 p = float3(0.0,0,0);
    [loop]
    for( int j=0; j< 105; j++ )
//    for( int j=0; j< 50; j++ )
    {
        if (p.y > 650.0 || t > 1300.0) break;
        p = rO + t*rD;
        float h = Map(p); // ...Get this positions height mapping.
        // Are we inside, and close enough to fudge a hit?...
        if( h < .5)
        {
            fin = true;
            distances = float2(t, oldT);
            break;
        }
                
        delta = clamp(0.5*h, .002, 20.0) + (t*0.004);
        oldT = t;
        t += delta;

        h = MapClouds(p);
        
        shade.y = max(-h, 0.0); 
        shade.x = smoothstep(.03, 1.5, shade.y);
        //shade.x = shade.x*shade.x;
        cloud += shade * (1.0 - cloud.y);


    }
    resT=0;
    if (fin) resT = BinarySubdivision(rO, rD, distances);
    
    cloud.x = 1.0-(cloud.x*5.3);//cloud.x = min(pow(max(cloud.x, 0.05), .3), 1.0);
 

    return fin;
}

//--------------------------------------------------------------------------
float3 CameraPath( float t )
{
    float m = 1.0+.5*300.0; //1.0+(iMouse.x/iResolution.x)*300.0;
    t =((_Time.y*2.0)+m+4005.0)*.004 + t;
    float2 p = 1500.0*float2( sin(4.5*t), cos(4.0*t) );
    return float3(-4800.0+p.x, 0.6, -200.0+p.y);
}

//--------------------------------------------------------------------------
// Some would say, most of the magic is done in post! :D
float3 PostEffects(float3 rgb, float2 uv)
{

    float rgb2=dot(rgb,float3(0.333,0.333,0.333));
    rgb = lerp( rgb, float3(rgb2,rgb2,rgb2), -1. );
    rgb = sqrt(rgb);
    rgb *= .5+0.5*pow(70.0*uv.x*(uv.y-.12)*(1.0-uv.x)*(.88-uv.y), 0.2 );
    //rgb = clamp(rgb+Hash12(rgb.rb+uv*_Time.y)*.1, 0.0, 1.0);
    return rgb;
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection*float3(2,1,2);                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz;                                             // ray origin

    float3 col  = fragColor;

    float2 xy = fragCoord.xy *.0001; // / iResolution.xy;
    float2 uv = (-1.0 + 2.0*xy) * float2(1,1); // float2(iResolution.x/iResolution.y,1.0);
    float3 camTar;
    

    // Use several forward heights, of decreasing influence with distance from the camera.

    cameraPos.xz = ro.xz+CameraPath(0.0).xz;
    camTar.xz    = ro.xz+CameraPath(.05).xz;
    camTar.y = cameraPos.y = ro.y+TerrainCam(CameraPath(0.0).xz) + 85.0;
    cameraPos.y +=  smoothstep(5.0, 0.0, _Time.y)*180.0;
    camTar.y -= camTar.y * .005;
    
    float roll = 0.2*sin(_Time.y*.3);
    float3 cw = normalize(camTar-cameraPos);
    float3 cp = float3(sin(roll), cos(roll),0.0);
    float3 cu = (cross(cw,cp));
    float3 cv = (cross(cu,cw));
    //float3 
//    rd = normalize( uv.x*cu + uv.y*cv + 1.4*cw );


    float distance;
    float2 cloud;
        col = GetSky(rd);

    if( !Scene(cameraPos,rd, uv, distance, cloud) )
    {
        // Missed scene, now just get the sky value...
        col = GetSky(rd);
        float sunAmount = max( dot( rd, sunLight), 0.0 );
        col = col + sunColour * pow(sunAmount, 8.0)*.8;
        col = col+ sunColour * min(pow(sunAmount, 800.0), .4);
    } 
    else
    {
        // Get world coordinate of landscape...
        float3 pos = cameraPos + distance * rd;
        // Get normal from sampling the high definition height map
        // Use the distance to sample larger gaps to help stop aliasing...
        float d = distance*distance;
        float p = min(4.0, .0001+.00002 * d);
        
        float3 nor    = float3(0.0,         Terrain2(pos.xz, d), 0.0);
        float3 v2     = nor-float3(p,       Terrain2(pos.xz+float2(p,0.0), d), 0.0);
        float3 v3     = nor-float3(0.0,     Terrain2(pos.xz+float2(0.0,-p), d), -p);
        nor = cross(v2, v3);
        nor = normalize(nor);

        // Get the colour using all available data...
        col = TerrainColour(pos, nor, distance);
    }
    float bri = pow(max(dot(rd, sunLight), 0.0), 24.0)*2.0;
    bri = ((cloud.y)) * bri;
//    col = lerp(col, float3(min(bri+cloud.x * float3(1.,.95, .9), 1.0)), min(cloud.y*(bri+1.0), 1.0));

//    col = PostEffects(min(col, 1.0), xy);
    
    

    fragColor=float4(col*smoothstep(0.0, 2.0, _Time.y) ,1.0);

                return fragColor;
            }

            ENDCG
        }
    }
}



