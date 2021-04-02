
Shader "Skybox/Storm"
{
    Properties
    {
        _MainTex ("tex3D", 3D) = "white" {}
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

            uniform sampler3D _MainTex; 
            float4 _SunDir,_XYZPos;

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


// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
#define PI 3.14159265359
#define maxDist 10.
#define nStep 30
#define nStepLight 3

float saturate(float i)
{
    return clamp(i,0.,1.);
}
float hash( float n ) {
    return frac(sin(n)*43758.5453);
}

//float hashi( uint n ) 
//{
//    // integer hash copied from Hugo Elias
//    n = (n << 13U) ^ n;
//    n = n * (n * n * 15731U + 789221U) + 1376312589U;
//    return float( n & uint3(0x7fffffffU,0x7fffffffU,0x7fffffffU))/float(0x7fffffff);
//}
float noise(float p){
    float fl = floor(p);
    float fc = frac(p);
    return lerp(hash(fl), hash(fl + 1.0), fc);
}

float noise (float3 x)
{
    //smoothing distance to texel
    x*=32.;
    x += 0.5;
    
    float3 i = floor(x);
    float3 f = frac(x);
    f = f*f*f*(f*(f*6.0-15.0)+10.0);
    x = f+i;    
    x-=0.5;
    
    return tex3D( _MainTex, x/32.0 ).x;
}

// Created by inigo quilez - iq/2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

const float3x3 m = float3x3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
                    -0.60, -0.48,  0.64 );


float fbm( float3 p ) { // in [0,1]
    float f;
    f  = 0.5000*noise( p ); p = mul(m,p*2.02);
    f += 0.2500*noise( p ); p = mul(m,p*2.03);
    f += 0.1250*noise( p ); p = mul(m,p*2.01);
    f += 0.0625*noise( p ); p = mul(m,p*2.04);
    f += 0.03125*noise( p );
    
    return f;
}
// --- End of: Created by inigo quilez --------------------


float3 camera (float2 ndc, float3 camPos, float f, float3 lookAt)
{
    float3 forward = normalize(lookAt - camPos);
    float3 right = cross(float3(0.,1,0.), forward);
    float3 up = normalize(cross (forward, right));
    right = normalize(cross (up, forward));
    
    float3 rd = up * ndc.y + right * ndc.x + f*forward;
    
    return rd;
}

float map (float3 p)
{
    float v;
    float3 location =p;
    
    p= p*0.009;
    
    v = fbm (p);
   
    v =  (noise(_Time.y*0.0025+p+0.15*(float3(v,v,v)))) ;
   
    float d =  saturate((1.-(length(location)/9.)));
    
    v=v*d-0.1;
    
    return 1.3*saturate(v);
}

float lightMarch(float3 ro, float3 lightPos)
{
    float3 rd = lightPos-ro;
    float d = length (rd);
    rd = rd/d;
    float t = 0.;
    float stepLength = d/ float(nStepLight);
    float densitySum = 0.;
    float sampleNoise;
    int i = 0;
    for (; i < nStepLight; i++)
    {
        sampleNoise = map ( ro + t * rd);
       
        densitySum += sampleNoise;
        
        t += stepLength;
    }
    
    return exp(- d * (densitySum / float(i)));
}

float3 calculateLight(float3 samplePos, float3 lightPos, float3 lightColor, float lightStr)
{
        float sampleLight = lightMarch (samplePos, lightPos);
        float distToLight = length(lightPos-samplePos)+1.;
        float3 light = lightColor * lightStr * (1./(distToLight*distToLight)) * sampleLight;

        return light;
}

float3 march(float3 ro, float3 rd, float dither, float var)
{
    float value = 0.;
    float t = dither;
    float densitySum = 0.;

    float stepLength = maxDist / float(nStep);
    float3 color = float3(0.01,0.02,0.05)*1.;
    for (int i = 0; i < nStep; i++)
    {
        
        float3 samplePos = ro + t * rd ; 
        float sampleNoise = map (samplePos);
        densitySum += sampleNoise;
        
        //light1
        float3 lightPos1 = float3 (-18,1.8,0);         
        float3 light1 = calculateLight(samplePos, lightPos1, float3 (0.6,0.25,0.15), 250.);
        
        //light2
        float3 lightPos2 = float3 (0.,0.,-15.);
        float3 light2 = calculateLight(samplePos, lightPos2, float3 (0.1 ,0.2,0.6), 200.);
        
        //light3
        float n = 1. * (noise(0.7*samplePos.y)-0.5)- 0.2*samplePos.y;
        float3 lightPos3 = float3 (n,samplePos.y,10.*(hash(floor(0.1*_Time.y))-0.5));     
        float storm =  lerp (1.2,0., sign(frac(-0.1+0.1*_Time.y)-0.15 )) * noise (20.*_Time.y);
        float3 light3 = calculateLight(samplePos, lightPos3, float3 (1.,1.,1.), storm);

        float3 ambientColor = float3 (.0,0.025,0.025);
        
        color += exp(- t*(densitySum/float(i+1)))  * sampleNoise * (ambientColor + light1 + light2 + light3);
        
        t +=  stepLength * var;
    }
    
   
    return color;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex3D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos;                                             // ray origin

 //   float2 uv = fragCoord/iResolution.xy;
 //   float2 ndc = uv * 2. - 1.;
 //   ndc.x *=iResolution.x/iResolution.y;
  
//    float2 mouse = iMouse.xy/iResolution.xy;
      
//    float3 lookAt = float3(0.);
    
    float distanceToCenter =  (sin(0.1*_Time.y)*0.5 + 1.) *3. +3.;
    float3 cameraPos = float3  (distanceToCenter*cos(0.15*_Time.y), 0., distanceToCenter*sin(00.15*_Time.y));
    cameraPos = float3 (10., 0., 10.);

    //float3 rd = camera(ndc, cameraPos, 1.0,lookAt);
    float var = length(rd)/1.0; //to get constant z in samples
    rd = normalize (rd);
    
    float dither = 0.5; //*hashi(uint(fragCoord.x+iResolution.x*fragCoord.y)+uint(iResolution.x*iResolution.y)*uint(iFrame));//Updated with iFrame dimension

    float3 col = march(cameraPos, rd, dither,var);
    //col.x= smoothstep(0.0,1.0,col.x);
    //col.y= smoothstep(0.0,1.0,col.y);
    //col.z= smoothstep(0.0,1.0,col.z);
    fragColor = float4(col,1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}




