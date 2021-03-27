Shader "Unlit/Storm"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv);
                // apply fog
                UNITY_APPLY_FOG(i.fogCoord, col);
                return col;
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
    return fract(sin(n)*43758.5453);
}

float hashi( uint n ) 
{
    // integer hash copied from Hugo Elias
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float( n & uvec3(0x7fffffffU))/float(0x7fffffff);
}
float noise(float p){
    float fl = floor(p);
    float fc = fract(p);
    return mix(hash(fl), hash(fl + 1.0), fc);
}

float noise (vec3 x)
{
    //smoothing distance to texel
    x*=32.;
    x += 0.5;
    
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f*f*f*(f*(f*6.0-15.0)+10.0);
    x = f+i;    
    x-=0.5;
    
    return texture( iChannel0, x/32.0 ).x;
}

// Created by inigo quilez - iq/2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

const mat3 m = mat3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
                    -0.60, -0.48,  0.64 );


float fbm( vec3 p ) { // in [0,1]
    float f;
    f  = 0.5000*noise( p ); p = m*p*2.02;
    f += 0.2500*noise( p ); p = m*p*2.03;
    f += 0.1250*noise( p ); p = m*p*2.01;
    f += 0.0625*noise( p ); p = m*p*2.04;
    f += 0.03125*noise( p );
    
    return f;
}
// --- End of: Created by inigo quilez --------------------


vec3 camera (vec2 ndc, vec3 camPos, float f, vec3 lookAt)
{
    vec3 forward = normalize(lookAt - camPos);
    vec3 right = cross(vec3(0.,1,0.), forward);
    vec3 up = normalize(cross (forward, right));
    right = normalize(cross (up, forward));
    
    vec3 rd = up * ndc.y + right * ndc.x + f*forward;
    
    return rd;
}

float map (vec3 p)
{
    float v;
    vec3 location =p;
    
    p= p*0.009;
    
    v = fbm (p);
   
    v =  (noise(iTime*0.0025+p+0.15*(vec3(v,v,v)))) ;
   
    float d =  saturate((1.-(length(location)/9.)));
    
    v=v*d-0.1;
    
    return 1.3*saturate(v);
}

float lightMarch(vec3 ro, vec3 lightPos)
{
    vec3 rd = lightPos-ro;
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

vec3 calculateLight(vec3 samplePos, vec3 lightPos, vec3 lightColor, float lightStr)
{
        float sampleLight = lightMarch (samplePos, lightPos);
        float distToLight = length(lightPos-samplePos)+1.;
        vec3 light = lightColor * lightStr * (1./(distToLight*distToLight)) * sampleLight;

        return light;
}

vec3 march(vec3 ro, vec3 rd, float dither, float var)
{
    float value = 0.;
    float t = dither;
    float densitySum = 0.;

    float stepLength = maxDist / float(nStep);
    vec3 color = vec3(0.01,0.02,0.05)*1.;
    for (int i = 0; i < nStep; i++)
    {
        
        vec3 samplePos = ro + t * rd ; 
        float sampleNoise = map (samplePos);
        densitySum += sampleNoise;
        
        //light1
        vec3 lightPos1 = vec3 (-18,1.8,0);         
        vec3 light1 = calculateLight(samplePos, lightPos1, vec3 (0.6,0.25,0.15), 250.);
        
        //light2
        vec3 lightPos2 = vec3 (0.,0.,-15.);
        vec3 light2 = calculateLight(samplePos, lightPos2, vec3 (0.1 ,0.2,0.6), 200.);
        
        //light3
        float n = 1. * (noise(0.7*samplePos.y)-0.5)- 0.2*samplePos.y;
        vec3 lightPos3 = vec3 (n,samplePos.y,10.*(hash(floor(0.1*iTime))-0.5));     
        float storm =  mix (1.2,0., sign(fract(-0.1+0.1*iTime)-0.15 )) * noise (20.*iTime);
        vec3 light3 = calculateLight(samplePos, lightPos3, vec3 (1.,1.,1.), storm);

        vec3 ambientColor = vec3 (.0,0.025,0.025);
        
        color += exp(- t*(densitySum/float(i+1)))  * sampleNoise * (ambientColor + light1 + light2 + light3);
        
        t +=  stepLength * var;
    }
    
   
    return color;
}



void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec2 ndc = uv * 2. - 1.;
    ndc.x *=iResolution.x/iResolution.y;
  
    vec2 mouse = iMouse.xy/iResolution.xy;
      
    vec3 lookAt = vec3(0.);
    
    float distanceToCenter =  (sin(0.1*iTime)*0.5 + 1.) *3. +3.;
    vec3 cameraPos = vec3  (distanceToCenter*cos(0.15*iTime), 0., distanceToCenter*sin(00.15*iTime));
    //cameraPos = vec3 (10.*mouse.y*cos(mouse.x*2.*PI), 0., 10.*mouse.y*sin(2.*PI*mouse.x ));

    vec3 rd = camera(ndc, cameraPos, 1.0,lookAt);
    float var = length(rd)/1.0; //to get constant z in samples
    rd = normalize (rd);
    
    float dither = 0.5*hashi(uint(fragCoord.x+iResolution.x*fragCoord.y)+uint(iResolution.x*iResolution.y)*uint(iFrame));//Updated with iFrame dimension

    vec3 col = march(cameraPos, rd, dither,var);
    //col.x= smoothstep(0.0,1.0,col.x);
    //col.y= smoothstep(0.0,1.0,col.y);
    //col.z= smoothstep(0.0,1.0,col.z);
    fragColor = vec4(col,1.0);
}


            ENDCG
        }
    }
}
