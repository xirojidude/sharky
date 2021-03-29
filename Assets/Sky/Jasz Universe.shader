
Shader "Skybox/JaszUniverse"
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

            //Generation settings
#define NOISE_ALPHA_MULTIPLIER 0.5
#define NOISE_SIZE_MULTIPLIER 1.8

//Uncomment to disable fog shape animation over time
#define MUTATE_SHAPE

//Rendering settings

//Uncoment to get high quality version (if you have good PC)
//#define HIGH_QUALITY

#ifdef HIGH_QUALITY
    #define RAYS_COUNT 150
    #define STEP_MODIFIER 1.007
    #define SHARPNESS 0.009
    #define NOISE_LAYERS_COUNT 5.0
    #define JITTERING 0.03
#else
    #define RAYS_COUNT 54
    #define STEP_MODIFIER 1.0175
    #define SHARPNESS 0.02
    #define NOISE_LAYERS_COUNT 4.0
    #define JITTERING 0.08
#endif

#define DITHER 0.3
#define NEAR_PLANE 0.6
#define RENDER_DISTANCE 2.0

//Colors
#define BRIGHTNESS 5.0
#define COLOR1 float3(0.0, 1.0, 1.0)
#define COLOR2 float3(1.0, 0.0, 0.9)

//Camera and time
#define TIME_SCALE 1.0
#define CAMERA_SPEED 0.04
#define CAMERA_ROTATION_SPEED 0.06
#define FOG_CHANGE_SPEED 0.02
//License: CC BY 3.0
//Author: Jan Mróz (jaszunio15)


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


float hash(float3 v)
{
    return frac(sin(dot(v, float3(11.51721, 67.12511, 9.7561))) * 1551.4172);   
}

float getNoiseFromfloat3(float3 v)
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

float volumetricFog(float3 v, float noiseMod)
{
    float noise = 0.0;
    float alpha = 1.0;
    float3 point1 = v;
    for(float i = 0.0; i < NOISE_LAYERS_COUNT; i++)
    {
        noise += getNoiseFromfloat3(point1) * alpha;
        point1 *= NOISE_SIZE_MULTIPLIER;
        alpha *= NOISE_ALPHA_MULTIPLIER;
    }
    
    //noise = noise / ((1.0 - pow(NOISE_ALPHA_MULTIPLIER, NOISE_LAYERS_COUNT))/(1.0 - NOISE_ALPHA_MULTIPLIER));
    noise *= 0.575;

    //edge + bloomy edge
#ifdef MUTATE_SHAPE
    float edge = 0.1 + getNoiseFromfloat3(v * 0.5 + float3(_Time.y * 0.03,_Time.y * 0.03,_Time.y * 0.03)) * 0.8;
#else
    float edge = 0.5;
#endif
    noise = (0.5 - abs(edge * (1.0 + noiseMod * 0.05) - noise)) * 2.0;
    return (smoothstep(1.0 - SHARPNESS * 2.0, 1.0 - SHARPNESS, noise * noise) + (1.0 - smoothstep(1.3, 0.6, noise))) * 0.2;
}


float3 nearPlanePoint(float2 v, float time)
{
    return float3(v.x, NEAR_PLANE * (1.0 + sin(time * 0.2) * 0.4), v.y);   
}

float3 fogMarch(float3 rayStart, float3 rayDirection, float time, float disMod)
{
    float stepLength = RENDER_DISTANCE / float(RAYS_COUNT);
    float3 fog = float3(0.0,0,0);   
    float3 point1 = rayStart;
    
    for(int i = 0; i < RAYS_COUNT; i++)
    {
        point1 += rayDirection *stepLength;
        fog += volumetricFog(point1, disMod) //intensity
            * lerp(COLOR1, COLOR2 * (1.0 + disMod * 0.5), getNoiseFromfloat3((point1 + float3(12.51, 52.167, 1.146)) * 0.5)) //coloring
            * lerp(1.0, getNoiseFromfloat3(point1 * 40.0) * 2.0, DITHER)    //Dithering
            * getNoiseFromfloat3(point1 * 0.2 + 20.0) * 2.0;   //Cutting big holes
        
        stepLength *= STEP_MODIFIER;
    }
    
    //There is a trick
    //Cutting mask in result, it will fake dynamic fog change, cover imperfections and add more 3D feeling
    fog = (fog / float(RAYS_COUNT)) * (pow(getNoiseFromfloat3((rayStart + rayDirection * RENDER_DISTANCE)), 2.0) * 3.0 + disMod * 0.5);
    
    return fog;
}

//Getting kick volume from spectrum
float getBeat()
{
    float sum = 0.0;
    for (float i = 0.0; i < 16.0; i++)
    {
        sum += tex2D(_MainTex, float2(i * 0.001 + 0.0, 0.0)).r;   
    }
    return smoothstep(0.6, 0.9, pow(sum * 0.06, 2.0));
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

    float time = _Time.y;
    float musicVolume = getBeat();
    float2 res = float2(800,450); //iResolution.xy;
    float2 uv = (2.0 * fragCoord - res) / res.x;
    
    //Random camera movement
    float3 cameraCenter = float3(sin(time * CAMERA_SPEED) * 10.0, time * CAMERA_SPEED * 10.0, cos(time * 0.78 * CAMERA_SPEED + 2.14) * 10.0);
    
    //Creating random rotation matrix for camera
    float angleY = sin(time * CAMERA_ROTATION_SPEED * 2.0);
    float angleX = cos(time * 0.712 * CAMERA_ROTATION_SPEED);
    float angleZ = sin(time * 1.779 * CAMERA_ROTATION_SPEED);
    float3x3 rotation =   float3x3(1, 0,            0,
                           0, sin(angleX),  cos(angleX),
                           0, -cos(angleX), sin(angleX))
                    * float3x3(sin(angleZ),  cos(angleZ), 0,
                           -cos(angleZ), sin(angleZ), 0,
                           0,            0,           1)
                    * float3x3(sin(angleY),  0, cos(angleY),
                           0,            1, 0,
                           -cos(angleY), 0, sin(angleY));
    
    float3 rayDirection = rd;// rotation * normalize(nearPlanePoint(uv, time));
    float3 rayStart = ro; //rayDirection * 0.2 + cameraCenter;  //Ray start with little clipping
    
    //Thanks to adx for jittering tip, looks and works really better with this line:
    rayStart += rayDirection * (hash(float3(uv + 4.0, frac(_Time.y) + 2.0)) - 0.5) * JITTERING;
    
    float3 fog = fogMarch(rayStart, rayDirection, time, musicVolume);
    
    //postprocess
    fog *= 2.5 * BRIGHTNESS;
    fog += 0.07 * lerp(COLOR1, COLOR2, 0.5); //Colouring the darkness
    fog = sqrt(smoothstep(0.0, 1.5, fog)); //Dealing with too bright areas (sometimes it happen)
    
    fragColor = float4(fog * smoothstep(0.0, 10.0, _Time.y), 1.0);

                return fragColor;
            }

            ENDCG
        }
    }
}




